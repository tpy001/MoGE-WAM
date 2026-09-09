#!/usr/bin/env python3
"""
Parallel LIBERO-Plus evaluation client optimized for throughput.

Main speedups:
1. Replan-only rendering for all categories except sensor noise:
   camera observables are enabled only when the action queue is empty.
2. Correct sensor-noise behavior:
   the noise category keeps normal observations enabled so its wrapper runs.
3. Multi-process task scheduler:
   multiple LIBERO environments run in parallel and share one policy server.
4. One task per job:
   all episodes of a task reuse the same LIBERO environment, avoiding repeated env creation.

Recommended:
    python eval_libero_plus_parallel_fast.py \
        --host 127.0.0.1 \
        --port 8000 \
        --client_type websocket \
        --task_suite libero_spatial \
        --category camera \
        --num_trials 1 \
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
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Deque, Dict

# LIBERO and LIBERO-Plus use the same import name. Put the Plus checkout and
# its config first, otherwise regular LIBERO can be imported silently.
LIBERO_PLUS_ROOT = Path(
    os.environ.get("LIBERO_PLUS_ROOT", "/data5/tpy/code/LIBERO-plus")
).resolve()
if LIBERO_PLUS_ROOT.is_dir():
    sys.path.insert(0, str(LIBERO_PLUS_ROOT))
    os.environ["LIBERO_CONFIG_PATH"] = os.environ.get(
        "LIBERO_PLUS_CONFIG_PATH", str(LIBERO_PLUS_ROOT / ".libero_config")
    )

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
from libero.libero.envs.env_wrapper import (
    fog,
    gaussian_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
)


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

CATEGORY_ALIASES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
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


def canonical_policy_prompt(task_description: str, category: str) -> str:
    """Remove LIBERO-Plus bookkeeping suffixes from the policy prompt."""
    patterns = {
        "camera": r"\s+view\s+.*$",
        "robot": r"\s+view\s+.*$",
        "light": r"\s+light\s+\d+\s*$",
        "background": r"\s+(?:table|tb)\s+\d+\s*$",
        "layout": r"\s+(?:add\s+\d+|level\d+\s+sample\d+)\s*$",
    }
    pattern = patterns.get(category.lower())
    return re.sub(pattern, "", task_description).strip() if pattern else task_description


def classification_rows(cfg: dict) -> list[dict]:
    category = CATEGORY_ALIASES.get(cfg["category"].lower(), cfg["category"])
    path = (
        Path(get_libero_path("benchmark_root"))
        / "benchmark"
        / "task_classification.json"
    )
    if not path.is_file():
        raise RuntimeError(
            f"LIBERO-Plus classification file not found: {path}. "
            "Set LIBERO_PLUS_ROOT and LIBERO_CONFIG_PATH to the "
            "LIBERO-Plus checkout."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        row
        for row in data[cfg["task_suite_name"]]
        if row["category"] == category
    ]
    rows = rows[cfg["task_start"] :]
    if cfg["task_limit"] is not None:
        rows = rows[: cfg["task_limit"]]
    return rows


def filtered_task_suite(cfg: dict):
    task_suite = benchmark_dict[cfg["task_suite_name"]]()
    selected_names = {row["name"] for row in classification_rows(cfg)}
    task_suite.tasks = [
        task for task in task_suite.tasks if task.name in selected_names
    ]
    task_suite.n_tasks = len(task_suite.tasks)
    missing = selected_names.difference(task.name for task in task_suite.tasks)
    if missing:
        raise RuntimeError(
            "Classification names missing from benchmark task map: "
            f"{sorted(missing)[:5]}"
        )
    return task_suite


def camera_observable_names(env) -> list[str]:
    observables = getattr(env.env, "_observables", {})
    return [name for name in observables if name.endswith("_image")]


def set_camera_observables_enabled(env, enabled: bool) -> None:
    for name in camera_observable_names(env):
        env.env._observables[name].set_enabled(enabled)


def apply_agentview_noise(env, obs: Dict) -> Dict:
    """Match LIBERO-Plus's sensor-noise postprocessing."""
    from PIL import Image

    img_array = obs["agentview_image"]
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array)
    if env.noise <= 10:
        blurred = motion_blur(pil_image, severity=env.noise)
    elif env.noise <= 20:
        blurred = gaussian_blur(pil_image, severity=env.noise - 10).astype(np.uint8)
    elif env.noise <= 30:
        blurred = zoom_blur(pil_image, severity=env.noise - 20).astype(np.uint8)
    elif env.noise <= 40:
        blurred = fog(pil_image, severity=env.noise - 30).astype(np.uint8)
    elif env.noise <= 50:
        blurred = glass_blur(pil_image, severity=env.noise - 40).astype(np.uint8)
    else:
        blurred = img_array
    obs["agentview_image"] = blurred
    return obs


def get_replan_observation(env) -> Dict:
    set_camera_observables_enabled(env, True)
    obs = env.env._get_observations(force_update=True)
    if env.noise != 0 and "agentview_image" in obs:
        obs = apply_agentview_noise(env, obs)
    return obs


def get_libero_env(task, resolution: int, seed: int):
    """
    Create one LIBERO environment.

    Camera observables are toggled at episode time.  The sensor-noise category
    keeps them enabled because its wrapper applies corruption during env.step().
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
    category: str,
    canonicalize_prompt: bool,
):
    """
    Run one episode.

    Important:
        render_cameras() is called ONLY when client.needs_replan() is True.
    """
    env.reset()
    client.reset()
    obs = env.set_init_state(initial_state)

    # Noise is injected by the LIBERO-Plus wrapper during normal observation
    # production, so keep camera observations enabled for that category.
    render_only_on_replan = category.lower() != "noise"
    if render_only_on_replan:
        set_camera_observables_enabled(env, False)

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
            if render_only_on_replan:
                obs = get_replan_observation(env)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
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
            if render_only_on_replan:
                set_camera_observables_enabled(env, False)
        else:
            # client.step() will consume an already-buffered action and
            # will not touch image / wrist_image.
            obs_dict = {
                "state": state,
            }

        action = np.asarray(
            client.step(
                obs_dict,
                canonical_policy_prompt(task_description, category)
                if canonicalize_prompt
                else task_description,
            ),
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
        task_suite = filtered_task_suite(cfg)
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

        num_trials = min(cfg["num_trials"], len(initial_states))
        for ep in range(num_trials):
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
                    category=cfg["category"],
                    canonicalize_prompt=cfg["canonicalize_prompt"],
                )

                ep_elapsed = time.perf_counter() - ep_start
                success = result["success"]

                if success:
                    successes += 1

                print(
                    f"[PID {worker_pid}] "
                    f"Task {task_id:03d} "
                    f"Ep {ep + 1:03d}/{num_trials:03d} "
                    f"{'SUCCESS' if success else 'FAIL'} "
                    f"steps={result['steps']} "
                    f"replans={result['num_replans']} "
                    f"time={ep_elapsed:.1f}s "
                    f"running={successes}/{ep + 1}",
                    flush=True,
                )
                
                keep_video = not cfg["save_failed_videos_only"] or not success
                if (
                    cfg["save_video"]
                    and video_path is not None
                    and video_path.exists()
                    and keep_video
                ):
                    final_path = video_path.with_name(
                        video_path.stem
                        + ("_success.mp4" if success else "_failure.mp4")
                    )
                    video_path.rename(final_path)
                elif cfg["save_video"] and video_path is not None and video_path.exists():
                    video_path.unlink()

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
            "num_trials": num_trials,
            "success_rate": successes / max(num_trials, 1),
            "elapsed_sec": elapsed,
            "worker_pid": worker_pid,
            "episodes": episode_results,
        }

    except Exception as exc:
        return {
            "task_id": task_id,
            "task_description": None,
            "successes": 0,
            "num_trials": 0,
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
    task_suite = filtered_task_suite(cfg)
    num_tasks = task_suite.n_tasks
    if num_tasks == 0:
        raise RuntimeError(
            "No LIBERO-Plus tasks matched the requested suite/category/slice"
        )

    task_ids = list(range(num_tasks - 1, -1, -1))

    print("=" * 78)
    print("Parallel LIBERO-Plus evaluation")
    print(f"  suite             : {cfg['task_suite_name']}")
    print(f"  category          : {cfg['category']}")
    print(f"  tasks             : {num_tasks}")
    print(f"  trials / task     : {cfg['num_trials']}")
    print(f"  workers           : {cfg['num_workers']}")
    print(f"  replan_steps      : {cfg['replan_steps']}")
    print(
        "  render            : "
        + ("every step (sensor noise)" if cfg["category"].lower() == "noise" else "replan only")
    )
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
                    "num_trials": 0,
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
        "Fast parallel LIBERO-Plus evaluation client"
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

    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Episodes per perturbed task (LIBERO-Plus normally uses 1).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--category",
        type=str,
        default="camera",
        choices=list(CATEGORY_ALIASES),
        help="LIBERO-Plus perturbation category.",
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=0,
        help="Start offset after category filtering.",
    )
    parser.add_argument(
        "--task_limit",
        type=int,
        default=None,
        help="Optional number of category-filtered tasks to evaluate.",
    )
    parser.add_argument(
        "--canonicalize_prompt",
        action="store_true",
        help="Strip perturbation bookkeeping suffixes from policy prompts.",
    )

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
        help="LIBERO environment camera resolution.",
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
        "--save_failed_videos_only",
        action="store_true",
        help="When saving videos, discard successful rollouts.",
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

    if args.num_trials != 1:
        raise ValueError("LIBERO-Plus evaluates each perturbed task exactly once; use --num_trials 1")

    if args.task_start < 0:
        raise ValueError("--task_start must be >= 0")

    if args.task_limit is not None and args.task_limit < 1:
        raise ValueError("--task_limit must be >= 1")

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
        "category": args.category,
        "task_start": args.task_start,
        "task_limit": args.task_limit,
        "canonicalize_prompt": args.canonicalize_prompt,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "env_resolution": args.env_resolution,
        "save_video": args.save_video,
        "save_failed_videos_only": args.save_failed_videos_only,
        "video_out_path": str(
            Path(args.video_out) / args.task_suite / args.category
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
