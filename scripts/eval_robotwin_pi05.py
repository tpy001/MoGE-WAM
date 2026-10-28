"""Serve trained pi05 weights or evaluate them with the official RoboTwin loop."""

# ruff: noqa: PLC0415 -- Server and simulator dependencies live in separate environments.

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def serve(cli):
    import jax
    from robotwin_pi05_policy import SessionPolicy

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    checkpoint = Path(cli.checkpoint).resolve()
    policy = policy_config.create_trained_policy(
        config.get_config(cli.config),
        checkpoint,
        sample_kwargs={"num_steps": cli.denoise_steps},
    )
    policy._rng = jax.random.key(cli.seed)  # noqa: SLF001 -- The factory does not expose its RNG argument.
    metadata = dict(policy.metadata, checkpoint=str(checkpoint), config=cli.config, seed=cli.seed)
    WebsocketPolicyServer(
        SessionPolicy(policy, lambda: jax.random.key(cli.seed)),
        host="127.0.0.1",
        port=cli.port,
        metadata=metadata,
        ping_interval=None,
        ping_timeout=None,
    ).serve_forever()


def task_args(cli, name, output):
    import yaml

    root = Path(cli.robotwin_root)
    args = yaml.safe_load((root / "task_config" / f"{cli.task_config}.yml").read_text())
    embodiments = yaml.safe_load((root / "task_config/_embodiment_config.yml").read_text())
    if args["embodiment"] != ["aloha-agilex"]:
        raise ValueError("pi05_robotwin_rgb was trained on the aloha-agilex 14-D robot")
    robot_file = root / embodiments["aloha-agilex"]["file_path"]
    robot = yaml.safe_load((robot_file / "config.yml").read_text())
    for camera in robot["static_camera_list"]:
        if camera["name"] == "head_camera":
            camera["type"] = cli.camera_type
    args["camera"].update(
        head_camera_type=cli.camera_type,
        wrist_camera_type=cli.camera_type,
        collect_head_camera=True,
        collect_wrist_camera=True,
    )
    camera_map = yaml.safe_load((root / "task_config/_camera_config.yml").read_text())
    camera_size = camera_map[cli.camera_type]
    args["data_type"].update(rgb=True, qpos=True, pointcloud=False, depth=False)
    args.update(
        task_name=name,
        task_config=cli.task_config,
        policy_name="robotwin_pi05_policy",
        ckpt_setting=cli.checkpoint,
        eval_mode=True,
        render_freq=0,
        save_data=False,
        save_path=str(output),
        eval_video_log=cli.video,
        head_camera_h=camera_size["h"],
        head_camera_w=camera_size["w"],
        left_robot_file=str(robot_file),
        right_robot_file=str(robot_file),
        left_embodiment_config=deepcopy(robot),
        right_embodiment_config=deepcopy(robot),
        dual_arm_embodied=True,
    )
    if cli.clear_cache_freq is not None:
        args["clear_cache_freq"] = cli.clear_cache_freq
    if cli.video and not cli.smoke_test:
        args["eval_video_save_dir"] = str(output)
    return args


def evaluate(cli):
    import numpy as np
    from PIL import Image
    import yaml

    if "ROBOTWIN_CUDA_MEMORY_FRACTION" in os.environ:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(float(os.environ["ROBOTWIN_CUDA_MEMORY_FRACTION"]), device=0)

    root = Path(cli.robotwin_root).resolve()
    cli.robotwin_root = str(root)
    cli.checkpoint = str(Path(cli.checkpoint).resolve())
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "description/utils"))
    from robotwin_pi05_policy import get_model
    from robotwin_pi05_policy import reset_model
    from robotwin_render_runtime import ObservationRenderRuntime
    from script.eval_policy import class_decorator

    # Local FastWAM loop: query expert validity before closing its scene.
    source = Path(__file__).with_name("robotwin_eval_loop.py")
    spec = importlib.util.spec_from_file_location("fastwam_robotwin_eval", source)
    fastwam_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fastwam_eval)

    from robotwin_seed_cache import fix_mplib_scene_reference
    from robotwin_seed_cache import quiet_step_progress

    quiet_step_progress()
    fix_mplib_scene_reference()

    def start_video(command, **kwargs):
        command = list(command)
        if Path(command[0]).name == "ffmpeg":
            command[1:1] = ["-filter_threads", "1", "-filter_complex_threads", "1"]
            command[-1:-1] = ["-threads", "1"]
        return subprocess.Popen(command, **kwargs)

    # Limit the vendored evaluator's video encoder without editing shared FastWAM.
    fastwam_eval.subprocess = SimpleNamespace(Popen=start_video, PIPE=subprocess.PIPE)

    task_names = cli.tasks
    if task_names == ["all"]:
        task_names = sorted(yaml.safe_load((root / "task_config/_eval_step_limit.yml").read_text()))
    model = get_model(vars(cli))
    results = []
    for name in task_names:
        random.seed(cli.seed)
        np.random.seed(cli.seed)
        task_output = output / name
        task_output.mkdir(exist_ok=True)
        args = task_args(cli, name, task_output)
        env = class_decorator(name)
        runtime = ObservationRenderRuntime(env, enabled=not (cli.legacy_render_sync or cli.render_every_step))
        action_trace = []
        chunk_trace = []
        if cli.record_actions:
            original_take_action = env.take_action

            def record_action(
                action, *positional, trace=action_trace, chunks=chunk_trace, take=original_take_action, **kwargs
            ):
                trace.append(np.asarray(action).copy())
                chunks.append(model.inference_calls)
                return take(action, *positional, **kwargs)

            env.take_action = record_action
        model.action_digest = hashlib.sha256()
        before_actions = model.executed_actions
        before_calls = model.inference_calls
        before_inference_seconds = model.inference_seconds
        started = time.monotonic()
        if cli.smoke_test:
            try:
                env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
                instruction = json.loads((root / "description/task_instruction" / f"{name}.json").read_text())
                env.set_instruction(instruction["full_description"])
                reset_model(model)
                observation = env.get_obs()
                for camera in ("head_camera", "left_camera", "right_camera"):
                    rgb = observation["observation"][camera]["rgb"]
                    if rgb.shape != (args["head_camera_h"], args["head_camera_w"], 3):
                        raise ValueError(f"Unexpected {camera} resolution: {rgb.shape}")
                    Image.fromarray(rgb).save(task_output / f"{camera}.png")
                for _ in range(2):
                    model.step(env, env.get_obs())
                state = np.asarray(env.get_obs()["joint_action"]["vector"])
                if not np.isfinite(state).all() or env.take_action_cnt != 2:
                    raise RuntimeError("Simulation did not execute two finite-state action steps")
                result = {
                    "task": name,
                    "smoke_test": True,
                    "executed_actions": env.take_action_cnt,
                    "inference_calls": model.inference_calls,
                    "state_shape": list(state.shape),
                    "camera_shape": list(observation["observation"]["head_camera"]["rgb"].shape),
                    "status": "PASS",
                }
            finally:
                env.close_env(clear_cache=True)
        else:
            # Keep RoboTwin's expert validation, seed filtering, instruction sampling and success checks.
            start_seed = 100000 * (1 + cli.seed)
            next_seed, successes = fastwam_eval.eval_policy(
                name,
                env,
                args,
                model,
                start_seed,
                test_num=cli.rollout_limit or cli.episodes,
                video_size=f"{args['head_camera_w']}x{args['head_camera_h']}" if cli.video else None,
                instruction_type=cli.instruction_type,
                skip_get_obs_within_replan=not cli.render_every_step,
            )
            result = {
                "task": name,
                "smoke_test": False,
                "episodes": cli.rollout_limit or cli.episodes,
                "successes": successes,
                "success_rate": successes / (cli.rollout_limit or cli.episodes),
                "start_seed": start_seed,
                "next_seed": next_seed,
            }
        result.update(
            elapsed_seconds=time.monotonic() - started,
            checkpoint=cli.checkpoint,
            config=cli.config,
            task_config=cli.task_config,
            instruction_type="full_description" if cli.smoke_test else cli.instruction_type,
            replan_steps=cli.replan_steps,
            denoise_steps=cli.denoise_steps,
            seed=cli.seed,
            camera_type=cli.camera_type,
            skip_get_obs_within_replan=not cli.render_every_step,
            evaluator=str(source),
            render_on_observation=runtime.enabled,
            runtime=runtime.stats,
            executed_actions=model.executed_actions - before_actions,
            inference_calls=model.inference_calls - before_calls,
            inference_seconds=model.inference_seconds - before_inference_seconds,
            action_sha256=model.action_digest.hexdigest(),
            seed_cache=None,
            lightweight_rollout=False,
            expert_check_mode="per_episode",
            expert_validity_before_close=True,
            clear_cache_freq=args["clear_cache_freq"],
        )
        if cli.record_actions:
            np.savez_compressed(
                task_output / "executed_actions.npz",
                absolute_actions=np.asarray(action_trace),
                inference_chunk=np.asarray(chunk_trace),
            )
        (task_output / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_robotwin_rgb")
    parser.add_argument("--serve", action="store_true", help="Run in the pi05 training environment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step directory containing params/ and assets/")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--robotwin-root", default="/data1/user10/robotwin/RoboTwin")
    parser.add_argument("--tasks", nargs="+", default=["adjust_bottle"], help="Task names, or all")
    parser.add_argument("--episodes", type=int, default=100, help="Expert-valid episodes per task")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", choices=["seen", "unseen"], default="unseen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, choices=range(1, 33), default=24)
    parser.add_argument("--fastwam-root", default="/data1/user10/fastwam/FastWAM")
    parser.add_argument("--camera-type", choices=["D435", "Large_D435"], default="D435")
    parser.add_argument(
        "--legacy-render-sync", action="store_true", help="Reference mode: synchronize renderer every physics step"
    )
    parser.add_argument("--render-every-step", action="store_true", help="Disable FastWAM observation skipping")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--record-actions", action="store_true")
    parser.add_argument(
        "--rollout-limit", type=int,
        help="Evaluate only this many expert-valid episodes.",
    )
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument("--lightweight-rollout", action="store_true")
    parser.add_argument("--clear-cache-freq", type=int)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Two real policy actions; not a success-rate evaluation"
    )
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results/robotwin_pi05"))
    cli = parser.parse_args()
    if cli.precheck_only or cli.seed_cache_dir or cli.lightweight_rollout:
        parser.error("Expert-cache replay is retired; use normal per-episode expert checking without cache flags")
    if cli.episodes < 1 or cli.denoise_steps < 1 or cli.seed < 0:
        parser.error("episodes and denoise-steps must be positive; seed must be nonnegative")
    if cli.rollout_limit is not None and not 1 <= cli.rollout_limit <= cli.episodes:
        parser.error("rollout-limit must be between 1 and episodes")
    if cli.clear_cache_freq is not None and cli.clear_cache_freq < 1:
        parser.error("clear-cache-freq must be positive")
    if not (Path(cli.checkpoint) / "params").is_dir() or not (Path(cli.checkpoint) / "assets").is_dir():
        parser.error("checkpoint must contain both params/ and assets/")
    logging.basicConfig(level=logging.INFO, force=True)
    serve(cli) if cli.serve else evaluate(cli)


if __name__ == "__main__":
    main()
