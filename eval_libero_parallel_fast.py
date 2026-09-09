#!/usr/bin/env python3
"""
Parallel LIBERO evaluation client optimized for throughput.

Main speedups:
1. use_camera_obs=False:
   env.step() does NOT render camera images every control step.
2. Replan-only rendering:
   agentview / wrist images are rendered only when the action queue is empty.
3. Multi-process task scheduler:
   multiple LIBERO environments run in parallel and share one policy server.
4. One task per job:
   all episodes of a task reuse the same LIBERO environment, avoiding repeated env creation.

Recommended:
    python eval_libero_parallel_fast.py \
        --host 127.0.0.1 \
        --port 8000 \
        --client_type websocket \
        --task_suite libero_10 \
        --num_trials 50 \
        --replan_steps 5 \
        --num_workers 4

For maximum speed, do NOT pass --save_video.

Notes:
- Each worker process owns its own MuJoCo / EGL context and its own network client.
- The policy server is shared by all workers.
- multiprocessing uses "spawn" to avoid fork-related MuJoCo/EGL/socket problems.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Deque, Dict

import imageio
import json_numpy
import numpy as np
import requests

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client

    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle."""
    quat = np.asarray(quat).copy()

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_state(obs: Dict) -> np.ndarray:
    """Build SimVLA 8D state vector."""
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32, copy=False)


def render_cameras(env, resolution: int = 256):
    """
    Render both cameras on demand.

    The 180-degree rotation is kept to match the original evaluation script.
    """
    img = env.sim.render(
        camera_name="agentview",
        width=resolution,
        height=resolution,
    )
    wrist_img = env.sim.render(
        camera_name="robot0_eye_in_hand",
        width=resolution,
        height=resolution,
    )

    img = np.ascontiguousarray(img[::-1, ::-1])
    wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1])

    return img, wrist_img


def get_libero_env(task, resolution: int, seed: int):
    """
    Create one LIBERO environment.

    Crucial optimization:
        use_camera_obs=False

    Therefore env.step() only advances simulation / low-dimensional observation.
    Camera rendering happens manually in render_cameras() at replanning time.
    """
    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "use_camera_obs": False,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


# -----------------------------------------------------------------------------
# Policy clients
# -----------------------------------------------------------------------------

class WebSocketClient:
    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int = 5,
        resize_size: int = 256,
    ):
        if not HAS_WS_CLIENT:
            raise ImportError(
                "openpi_client not installed. Run: pip install openpi-client"
            )

        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def needs_replan(self) -> bool:
        return len(self.action_plan) == 0

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if self.needs_replan():
            # Images must only exist on a replan step.
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    obs["image"], self.resize_size, self.resize_size
                )
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    obs["wrist_image"], self.resize_size, self.resize_size
                )
            )

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }

            result = self.client.infer(element)
            action_chunk = np.asarray(result["actions"])

            if len(action_chunk) < self.replan_steps:
                raise ValueError(
                    f"Need at least {self.replan_steps} actions, "
                    f"but server returned {len(action_chunk)}"
                )

            for action in action_chunk[: self.replan_steps]:
                self.action_plan.append(np.asarray(action))

        return self.action_plan.popleft()


class HTTPClient:
    def __init__(self, host: str, port: int, replan_steps: int = 5):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def needs_replan(self) -> bool:
        return len(self.action_plan) == 0

    def infer(self, element: Dict) -> Dict:
        payload = {
            "image0": json_numpy.dumps(element["observation/image"]),
            "image1": json_numpy.dumps(element["observation/wrist_image"]),
            "proprio": json_numpy.dumps(element["observation/state"]),
            "language_instruction": element["prompt"],
            "steps": max(10, self.replan_steps),
        }

        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        return {"actions": np.asarray(data["action"])}

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if self.needs_replan():
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }

            result = self.infer(element)
            action_chunk = np.asarray(result["actions"])

            if len(action_chunk) < self.replan_steps:
                raise ValueError(
                    f"Need at least {self.replan_steps} actions, "
                    f"but server returned {len(action_chunk)}"
                )

            for action in action_chunk[: self.replan_steps]:
                self.action_plan.append(np.asarray(action))

        return self.action_plan.popleft()


def make_client(cfg: dict):
    if cfg["client_type"] == "websocket":
        return WebSocketClient(
            cfg["host"],
            cfg["port"],
            replan_steps=cfg["replan_steps"],
            resize_size=cfg["image_size"],
        )

    return HTTPClient(
        cfg["host"],
        cfg["port"],
        replan_steps=cfg["replan_steps"],
    )


# -----------------------------------------------------------------------------
# Single episode
# -----------------------------------------------------------------------------

def run_one_episode(
    env,
    client,
    task_description: str,
    initial_state,
    max_steps: int,
    resolution: int,
    save_video: bool,
    video_path: Path | None,
    video_fps: int,
):
    """
    Run one episode.

    Important:
        render_cameras() is called ONLY when client.needs_replan() is True.
    """
    env.reset()
    client.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    t = 0
    done = False
    num_replans = 0

    while t < max_steps + NUM_STEPS_WAIT:
        # Let objects stabilize. No image rendering here.
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        state = build_state(obs)

        if client.needs_replan():
            # ---------------------------------------------
            # ONLY render on a true replanning step.
            # ---------------------------------------------
            img, wrist_img = render_cameras(env, resolution)
            num_replans += 1

            obs_dict = {
                "image": img,
                "wrist_image": wrist_img,
                "state": state,
            }

            # Video is intentionally sparse: only replan frames.
            # This preserves the rendering speedup.
            if save_video:
                replay_images.append(img)
        else:
            # client.step() will consume an already-buffered action and
            # will not touch image / wrist_image.
            obs_dict = {
                "state": state,
            }

        action = np.asarray(
            client.step(obs_dict, task_description),
            dtype=np.float32,
        )

        if action.shape != (7,):
            raise ValueError(
                f"Expected relative action shape (7,), got {action.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError("Policy action contains NaN or Inf")

        obs, reward, done, info = env.step(action.tolist())

        if done:
            break

        t += 1

    if save_video and replay_images and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            str(video_path),
            replay_images,
            fps=video_fps,
        )

    return {
        "success": bool(done),
        "steps": int(t),
        "num_replans": int(num_replans),
    }


# -----------------------------------------------------------------------------
# Process worker: one task is one scheduling job
# -----------------------------------------------------------------------------

def evaluate_task_worker(cfg: dict, task_id: int) -> dict:
    """
    Evaluate all episodes of one LIBERO task.

    This function runs inside a spawned worker process.

    Why task-level scheduling?
    - The env is created only once for all episodes of this task.
    - ProcessPoolExecutor dynamically gives the next task to whichever worker
      becomes free first.
    """
    worker_pid = os.getpid()
    task_start = time.perf_counter()

    env = None

    try:
        # Each process creates its own benchmark object, client and env.
        task_suite = benchmark_dict[cfg["task_suite_name"]]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        client = make_client(cfg)

        env_seed = cfg["seed"] + task_id * 1000
        env, task_description = get_libero_env(
            task,
            cfg["env_resolution"],
            env_seed,
        )

        successes = 0
        episode_results = []

        for ep in range(cfg["num_trials"]):
            suffix_base = task_description.replace(" ", "_")[:50]

            video_path = None
            if cfg["save_video"]:
                # Suffix is finalized after the episode, so first write to
                # a temporary task/episode filename and rename below.
                video_path = (
                    Path(cfg["video_out_path"])
                    / f"task{task_id:03d}_{suffix_base}_ep{ep:03d}.mp4"
                )

            ep_start = time.perf_counter()

            try:
                result = run_one_episode(
                    env=env,
                    client=client,
                    task_description=task_description,
                    initial_state=initial_states[ep % len(initial_states)],
                    max_steps=cfg["max_steps"],
                    resolution=cfg["env_resolution"],
                    save_video=cfg["save_video"],
                    video_path=video_path,
                    video_fps=cfg["video_fps"],
                )

                ep_elapsed = time.perf_counter() - ep_start
                success = result["success"]

                if success:
                    successes += 1

                print(
                    f"[PID {worker_pid}] "
                    f"Task {task_id:03d} "
                    f"Ep {ep + 1:03d}/{cfg['num_trials']:03d} "
                    f"{'SUCCESS' if success else 'FAIL'} "
                    f"steps={result['steps']} "
                    f"replans={result['num_replans']} "
                    f"time={ep_elapsed:.1f}s "
                    f"running={successes}/{ep + 1}",
                    flush=True,
                )
                
                if cfg["save_video"] and video_path is not None and video_path.exists():
                    final_path = video_path.with_name(
                        video_path.stem
                        + ("_success.mp4" if success else "_failure.mp4")
                    )
                    video_path.rename(final_path)

                episode_results.append(
                    {
                        "episode": ep,
                        "success": success,
                        "steps": result["steps"],
                        "num_replans": result["num_replans"],
                        "elapsed_sec": ep_elapsed,
                    }
                )

            except Exception as exc:
                ep_elapsed = time.perf_counter() - ep_start

                episode_results.append(
                    {
                        "episode": ep,
                        "success": False,
                        "steps": None,
                        "num_replans": None,
                        "elapsed_sec": ep_elapsed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

                print(
                    f"[PID {worker_pid}] "
                    f"Task {task_id} Ep {ep} ERROR: {exc}",
                    flush=True,
                )

        elapsed = time.perf_counter() - task_start

        return {
            "task_id": task_id,
            "task_description": task_description,
            "successes": successes,
            "num_trials": cfg["num_trials"],
            "success_rate": successes / max(cfg["num_trials"], 1),
            "elapsed_sec": elapsed,
            "worker_pid": worker_pid,
            "episodes": episode_results,
        }

    except Exception as exc:
        return {
            "task_id": task_id,
            "task_description": None,
            "successes": 0,
            "num_trials": cfg["num_trials"],
            "success_rate": 0.0,
            "elapsed_sec": time.perf_counter() - task_start,
            "worker_pid": worker_pid,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "episodes": [],
        }

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Parallel scheduler
# -----------------------------------------------------------------------------

def eval_libero_parallel(cfg: dict) -> float:
    task_suite = benchmark_dict[cfg["task_suite_name"]]()
    num_tasks = task_suite.n_tasks

    task_ids = list(range(num_tasks - 1, -1, -1))

    print("=" * 78)
    print("Parallel LIBERO evaluation")
    print(f"  suite             : {cfg['task_suite_name']}")
    print(f"  tasks             : {num_tasks}")
    print(f"  trials / task     : {cfg['num_trials']}")
    print(f"  workers           : {cfg['num_workers']}")
    print(f"  replan_steps      : {cfg['replan_steps']}")
    print(f"  camera obs        : disabled during env.step()")
    print(f"  render            : replan only")
    print(f"  save video        : {cfg['save_video']}")
    print(
        f"  server            : "
        f"{'ws' if cfg['client_type'] == 'websocket' else 'http'}://"
        f"{cfg['host']}:{cfg['port']}"
    )
    print("=" * 78)

    wall_start = time.perf_counter()

    results = []

    # Explicit spawn context is important for MuJoCo / EGL / sockets.
    ctx = mp.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cfg["num_workers"],
        mp_context=ctx,
    ) as executor:
        future_to_task = {
            executor.submit(evaluate_task_worker, cfg, task_id): task_id
            for task_id in task_ids
        }

        completed = 0

        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            completed += 1

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "task_description": None,
                    "successes": 0,
                    "num_trials": cfg["num_trials"],
                    "success_rate": 0.0,
                    "elapsed_sec": 0.0,
                    "worker_pid": None,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "episodes": [],
                }

            results.append(result)

            if "fatal_error" in result:
                print(
                    f"[{completed:02d}/{num_tasks:02d}] "
                    f"Task {task_id:03d} FATAL: {result['fatal_error']}",
                    flush=True,
                )
            else:
                print(
                    f"[{completed:02d}/{num_tasks:02d}] "
                    f"Task {task_id:03d} "
                    f"{result['successes']}/{result['num_trials']} "
                    f"({result['success_rate'] * 100:5.1f}%) "
                    f"{result['elapsed_sec']:.1f}s "
                    f"PID={result['worker_pid']}",
                    flush=True,
                )

    wall_elapsed = time.perf_counter() - wall_start

    results.sort(key=lambda x: x["task_id"])

    total_successes = sum(r["successes"] for r in results)
    total_episodes = sum(r["num_trials"] for r in results)
    success_rate = total_successes / max(total_episodes, 1)

    total_replans = 0
    completed_episodes = 0
    for task_result in results:
        for ep_result in task_result.get("episodes", []):
            if ep_result.get("num_replans") is not None:
                total_replans += ep_result["num_replans"]
                completed_episodes += 1

    episodes_per_sec = total_episodes / max(wall_elapsed, 1e-9)

    print("\n" + "=" * 78)
    print("Final summary")
    print(f"  success           : {total_successes}/{total_episodes}")
    print(f"  success rate      : {success_rate * 100:.2f}%")
    print(f"  wall time         : {wall_elapsed:.2f}s")
    print(f"  throughput        : {episodes_per_sec:.4f} episodes/s")
    if completed_episodes:
        print(
            f"  avg replans / ep  : "
            f"{total_replans / completed_episodes:.2f}"
        )
    print("=" * 78)

    summary = {
        "config": cfg,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": success_rate,
        "wall_elapsed_sec": wall_elapsed,
        "episodes_per_sec": episodes_per_sec,
        "results": results,
    }

    output_json = Path(cfg["result_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results written to: {output_json}")

    return success_rate


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        "Fast parallel LIBERO evaluation client"
    )

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--connection_info",
        type=str,
        default=None,
        help="Path to server connection info JSON",
    )

    parser.add_argument(
        "--client_type",
        type=str,
        default="websocket",
        choices=["websocket", "http"],
    )

    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
    )

    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--replan_steps",
        type=int,
        default=5,
        help="Number of buffered actions executed before next image render/inference.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel LIBERO processes / environments.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Client-side model image resize.",
    )

    parser.add_argument(
        "--env_resolution",
        type=int,
        default=LIBERO_ENV_RESOLUTION,
        help="Resolution used by manual MuJoCo camera rendering.",
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help=(
            "Save sparse videos containing only replan frames. "
            "Disabled by default for maximum speed."
        ),
    )

    parser.add_argument(
        "--video_out",
        type=str,
        default="./eval_results_parallel/videos",
    )

    parser.add_argument(
        "--video_fps",
        type=int,
        default=10,
        help="FPS for sparse replan-frame videos.",
    )

    parser.add_argument(
        "--result_json",
        type=str,
        default="./eval_results_parallel/results.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    if args.replan_steps < 1:
        raise ValueError("--replan_steps must be >= 1")

    # Optional server connection-info file.
    if args.connection_info:
        print(f"Loading connection info: {args.connection_info}")

        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server connection info...")
            sys.stdout.flush()
            time.sleep(0.5)

        print()

        with open(args.connection_info, "r", encoding="utf-8") as f:
            info = json.load(f)

        args.host = info["host"]
        args.port = info["port"]

    max_steps = MAX_STEPS.get(args.task_suite, 400)

    cfg = {
        "host": args.host,
        "port": args.port,
        "client_type": args.client_type,
        "task_suite_name": args.task_suite,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "env_resolution": args.env_resolution,
        "save_video": args.save_video,
        "video_out_path": str(
            Path(args.video_out) / args.task_suite
        ),
        "video_fps": args.video_fps,
        "result_json": args.result_json,
        "max_steps": max_steps,
    }

    eval_libero_parallel(cfg)


if __name__ == "__main__":
    # Do not use fork with MuJoCo / EGL / sockets.
    mp.set_start_method("spawn", force=True)
    main()
