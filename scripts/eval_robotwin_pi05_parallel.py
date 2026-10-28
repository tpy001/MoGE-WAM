"""FastWAM-style process scheduling with persistent, per-GPU pi05 servers."""

import argparse
from collections import deque
import contextlib
import csv
import functools
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pynvml
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/eval_robotwin_pi05.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_rgb/robotwin_rgb_restored_20261019_201221/29999"
TASK_LIST = Path("/data1/user10/robotwin/RoboTwin/task_config/_eval_step_limit.yml")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", choices=range(8), default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tasks", nargs="+", default=["adjust_bottle", "lift_pot"])
    parser.add_argument("--phases", nargs="+", choices=["clean", "random"], default=["clean", "random"])
    parser.add_argument("--config", default="pi05_robotwin_rgb")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "evaluation_results" / time.strftime("pi05_parallel_%Y%m%d_%H%M%S")
    )
    parser.add_argument("--base-port", type=int, default=18780)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, choices=range(1, 33), default=24)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--legacy-render-sync", action="store_true")
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument("--lightweight-rollout", action="store_true")
    parser.add_argument("--clear-cache-freq", type=int)
    parser.add_argument("--resume", action="store_true", help="Keep completed task/phase results in output-dir")
    parser.add_argument("--min-free-gpu-mib", type=int, default=6144)
    parser.add_argument("--client-reserve-mib", type=int, default=10240)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.precheck_only or args.seed_cache_dir or args.lightweight_rollout:
        parser.error("Expert-cache replay is retired; omit cache flags to run normal per-episode expert checking")
    if len(args.gpus) != len(set(args.gpus)) or len(args.gpus) > 2:
        parser.error("Select one or two unique physical GPUs from 0-7")
    if args.episodes < 1 or args.denoise_steps < 1 or args.seed < 0:
        parser.error("episodes/denoise-steps must be positive and seed must be nonnegative")
    minimum_client_reserve = 2048 if args.lightweight_rollout else 6144
    if args.min_free_gpu_mib < 4096 or args.client_reserve_mib < minimum_client_reserve:
        parser.error(
            "GPU floor must be >=4096 MiB; client reservation must be >=2048 MiB for lightweight rollout, >=6144 otherwise"
        )
    if args.lightweight_rollout and (args.precheck_only or not args.seed_cache_dir):
        parser.error("Lightweight rollout requires a precomputed expert-seed cache")
    if args.tasks == ["all"]:
        args.tasks = list(yaml.safe_load(TASK_LIST.read_text()))
        if len(args.tasks) != 50:
            parser.error("Expected the 50-task FastWAM RoboTwin benchmark")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.phases)) != len(args.phases):
        parser.error("Duplicate tasks/phases would overwrite evaluation outputs")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def process_env(gpu):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{ROOT}/src:{ROOT}/packages/openpi-client/src",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        OPENCV_FOR_THREADS_NUM="1",
        ROBOTWIN_CUDA_MEMORY_FRACTION="0.16",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.30",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json",
        CUDA_HOME="/usr/local/cuda-12.4",
        TMPDIR="/data1/user10/robotwin/tmp",
        TORCH_EXTENSIONS_DIR="/data1/user10/robotwin/cache/torch_extensions",
    )
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env["PATH"]
    return env


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@functools.cache
def gpu_handle(gpu):
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(gpu)


def gpu_free_mib(gpu):
    return pynvml.nvmlDeviceGetMemoryInfo(gpu_handle(gpu)).free // (1024 * 1024)


def completed_result(path, args):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if getattr(args, "precheck_only", False):
        if result.get("mode") != "precheck" or result["valid_seeds"] != args.episodes:
            raise ValueError(f"Incomplete expert precheck result: {path}")
        return result
    for key in ("checkpoint", "episodes", "seed", "replan_steps", "denoise_steps"):
        if result[key] != (str(args.checkpoint) if key == "checkpoint" else getattr(args, key)):
            raise ValueError(f"Cannot resume mismatched {key}: {path}")
    if result["render_on_observation"] != (not args.legacy_render_sync):
        raise ValueError(f"Cannot resume different rendering mode: {path}")
    if not 0 <= result["successes"] <= args.episodes:
        raise ValueError(f"Invalid success count: {path}")
    return result


def check_thread_budget():
    limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    if limit == resource.RLIM_INFINITY:
        return None
    used = 0
    uid = os.getuid()
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
            if int(fields["Uid"].split()[0]) == uid:
                used += int(fields["Threads"])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if used >= limit - 128:
        raise RuntimeError(f"User thread budget exhausted: {used}/{limit}; stopping this evaluation")
    return used


def main(args):
    jobs = [(task, phase) for task in args.tasks for phase in args.phases]
    plan = {
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "jobs": jobs,
        "task_count": len(args.tasks),
        "total_episodes": len(jobs) * args.episodes,
        "episodes": args.episodes,
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "denoise_steps": args.denoise_steps,
        "render_on_observation": not args.legacy_render_sync,
        "mode": "precheck" if args.precheck_only else "evaluate",
        "seed_cache_dir": str(args.seed_cache_dir) if args.seed_cache_dir else None,
        "lightweight_rollout": args.lightweight_rollout,
        "clear_cache_freq": args.clear_cache_freq,
        "reserved_gpus": [],
        "unused_gpus": [gpu for gpu in range(8) if gpu not in args.gpus],
        "video": args.video,
        "min_free_gpu_mib": args.min_free_gpu_mib,
        "client_reserve_mib": args.client_reserve_mib,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets").is_dir():
        raise ValueError("checkpoint must contain params/ and assets/")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    config_file = args.output_dir / "run_config.json"
    if args.resume and config_file.exists():
        previous = json.loads(config_file.read_text())
        for key in (
            "jobs",
            "episodes",
            "checkpoint",
            "seed",
            "replan_steps",
            "denoise_steps",
            "render_on_observation",
        ):
            if previous[key] != json.loads(json.dumps(plan[key])):
                raise ValueError(f"Resume protocol differs: {key}")
    (args.output_dir / "run_config.json").write_text(json.dumps(plan, indent=2))
    started = time.monotonic()
    servers, running, files, results = [], [], [], []
    queue = deque()
    for task, phase in jobs:
        path = args.output_dir / f"{task}_{phase}" / task / "result.json"
        result = completed_result(path, args) if args.resume else None
        if result is None:
            queue.append((task, phase))
        else:
            if result["task"] != task or result["task_config"] != (
                "demo_clean" if phase == "clean" else "demo_randomized"
            ):
                raise ValueError(f"Task/phase mismatch: {path}")
            result.update(phase=phase, gpu=args.gpus[0])
            results.append(result)
    resources = {"free_gpu_mib": {}, "minimum_observed_free_gpu_mib": {}}
    last_resource_check = 0
    status = "running"
    resource_retries = {}

    def launch(command, gpu, log):
        stream = log.open("w")
        files.append(stream)
        return subprocess.Popen(
            command,
            cwd=ROOT,
            env=process_env(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def write_summary():
        summary = {
            "status": status,
            "wall_seconds": time.monotonic() - started,
            "total_evaluated": 0 if args.precheck_only else sum(result["episodes"] for result in results),
            "total_validated_seeds": sum(result["valid_seeds"] for result in results) if args.precheck_only else 0,
            "mode": "precheck" if args.precheck_only else "evaluate",
            "target_episodes": len(jobs) * args.episodes,
            "completed_jobs": len(results),
            "resources": resources,
            "results": results,
            "running": [
                {"task": r["task"], "phase": r["phase"], "gpu": r["gpu"], "pid": r["process"].pid} for r in running
            ],
            "pending": list(queue),
        }
        temporary = args.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2))
        temporary.replace(args.output_dir / "summary.json")
        with (args.output_dir / "summary.csv").open("w", newline="") as stream:
            columns = ["task", "phase", "gpu", "episodes", "successes", "success_rate", "elapsed_seconds"]
            if args.precheck_only:
                columns = ["task", "phase", "gpu", "valid_seeds", "seed_attempts", "elapsed_seconds"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: result[key] for key in columns} for result in results)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_summary()
    try:
        check_thread_budget()
        if not queue:
            status = "complete"
            return
        for index, gpu in enumerate(args.gpus):
            if args.precheck_only:
                servers.append((gpu, args.base_port + index, None))
                continue
            if gpu_free_mib(gpu) < args.min_free_gpu_mib + args.client_reserve_mib + 8192:
                raise RuntimeError(f"GPU {gpu} has insufficient free memory for server plus client")
            port = args.base_port + index
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            cpu_offset = int(os.environ.get("ROBOTWIN_EVAL_CPU_OFFSET", "0"))
            cpu = f"{48 + cpu_offset + index * 4}-{51 + cpu_offset + index * 4}"
            command = [
                "taskset",
                "-c",
                cpu,
                str(ROOT / ".venv/bin/python"),
                str(ENTRY),
                "--serve",
                "--config",
                args.config,
                "--checkpoint",
                str(args.checkpoint),
                "--port",
                str(port),
                "--seed",
                str(args.seed),
                "--denoise-steps",
                str(args.denoise_steps),
            ]
            process = launch(command, gpu, args.output_dir / f"server_gpu{gpu}.log")
            servers.append((gpu, port, process))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for gpu, port, process in servers:
            if process is None:
                continue
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Policy GPU {gpu} exited; see server_gpu{gpu}.log")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Policy server GPU {gpu} did not become ready") from None
                    time.sleep(1)
        while queue or running:
            if time.monotonic() - last_resource_check >= 5:
                resources["user_threads"] = check_thread_budget()
                resources["maximum_observed_user_threads"] = max(
                    resources.get("maximum_observed_user_threads", 0), resources["user_threads"] or 0
                )
                for gpu in args.gpus:
                    free = gpu_free_mib(gpu)
                    resources["free_gpu_mib"][gpu] = free
                    low = resources["minimum_observed_free_gpu_mib"].get(gpu, free)
                    resources["minimum_observed_free_gpu_mib"][gpu] = min(low, free)
                    if free < args.min_free_gpu_mib:
                        raise RuntimeError(
                            f"GPU {gpu} free memory {free} MiB below safety floor; resume with one client"
                        )
                last_resource_check = time.monotonic()
            for index, (gpu, port, _) in enumerate(servers):
                available = resources["free_gpu_mib"][gpu]
                # Reserve startup allocations that nvidia-smi may not yet show.
                available -= args.client_reserve_mib * sum(
                    r["gpu"] == gpu and time.monotonic() - r["started"] < 60 for r in running
                )
                occupied = {r["slot"] for r in running if r["gpu"] == gpu}
                for slot in range(args.workers_per_gpu):
                    if not queue:
                        break
                    if slot in occupied:
                        continue
                    if not running and gpu_free_mib(gpu) < args.min_free_gpu_mib + args.client_reserve_mib:
                        raise RuntimeError("Insufficient memory to start a client; completed results are retained")
                    if available < args.min_free_gpu_mib + args.client_reserve_mib:
                        continue
                    task, phase = queue.popleft()
                    job_output = args.output_dir / f"{task}_{phase}"
                    if job_output.exists():
                        archive = args.output_dir / "incomplete_attempts"
                        archive.mkdir(exist_ok=True)
                        job_output.rename(archive / f"{task}_{phase}_{time.time_ns()}")
                    job_output.mkdir()
                    cpu_start = 56 + int(os.environ.get("ROBOTWIN_EVAL_CPU_OFFSET", "0")) + (index * args.workers_per_gpu + slot) * 4
                    command = [
                        "taskset",
                        "-c",
                        f"{cpu_start}-{cpu_start + 3}",
                        "/data1/user10/robotwin/env/bin/python",
                        str(ENTRY),
                        "--config",
                        args.config,
                        "--checkpoint",
                        str(args.checkpoint),
                        "--port",
                        str(port),
                        "--tasks",
                        task,
                        "--task-config",
                        "demo_clean" if phase == "clean" else "demo_randomized",
                        "--episodes",
                        str(args.episodes),
                        "--seed",
                        str(args.seed),
                        "--replan-steps",
                        str(args.replan_steps),
                        "--denoise-steps",
                        str(args.denoise_steps),
                        "--output-dir",
                        str(job_output),
                    ]
                    if args.video:
                        command.append("--video")
                    if args.precheck_only:
                        command.append("--precheck-only")
                    if args.seed_cache_dir:
                        command.extend(["--seed-cache-dir", str(args.seed_cache_dir)])
                    if args.lightweight_rollout:
                        command.append("--lightweight-rollout")
                    if args.clear_cache_freq is not None:
                        command.extend(["--clear-cache-freq", str(args.clear_cache_freq)])
                    if args.legacy_render_sync:
                        command.append("--legacy-render-sync")
                    process = launch(command, gpu, job_output / "eval.log")
                    running.append(
                        {
                            "task": task,
                            "phase": phase,
                            "gpu": gpu,
                            "slot": slot,
                            "process": process,
                            "output": job_output,
                            "started": time.monotonic(),
                        }
                    )
                    resources["maximum_running_clients"] = max(
                        resources.get("maximum_running_clients", 0), len(running)
                    )
                    available -= args.client_reserve_mib
                    print(f"launch {task}/{phase} GPU={gpu} pid={process.pid}", flush=True)
            write_summary()
            for worker in list(running):
                code = worker["process"].poll()
                if code is None:
                    continue
                if code:
                    key = (worker["task"], worker["phase"])
                    log_tail = (worker["output"] / "eval.log").read_text(errors="replace")[-8192:]
                    retries = resource_retries.get(key, 0)
                    if "RuntimeError: Resource temporarily unavailable" in log_tail and retries < 2:
                        resource_retries[key] = retries + 1
                        resources.setdefault("resource_retries", {})["/".join(key)] = retries + 1
                        running.remove(worker)
                        queue.appendleft(key)
                        print(f"retry {key[0]}/{key[1]} after renderer resource error ({retries + 1}/2); restarting full task", flush=True)
                        continue
                    raise RuntimeError(
                        f"Worker failed: {worker['task']}/{worker['phase']}; see {worker['output']}/eval.log"
                    )
                result = json.loads((worker["output"] / worker["task"] / "result.json").read_text())
                result.update(phase=worker["phase"], gpu=worker["gpu"])
                results.append(result)
                running.remove(worker)
                score = (
                    f"{result['valid_seeds']} expert-valid seeds"
                    if args.precheck_only
                    else f"{result['successes']}/{result['episodes']}"
                )
                print(f"done {result['task']}/{result['phase']}: {score}", flush=True)
            time.sleep(1)
        status = "complete"
    except BaseException:
        status = "failed"
        raise
    finally:
        for worker in running:
            stop_process(worker["process"])
        for _, _, process in servers:
            if process is not None:
                stop_process(process)
        for stream in files:
            stream.close()
        write_summary()
        with (args.output_dir / "summary.csv").open("w", newline="") as stream:
            columns = ["task", "phase", "gpu", "episodes", "successes", "success_rate", "elapsed_seconds"]
            if args.precheck_only:
                columns = ["task", "phase", "gpu", "valid_seeds", "seed_attempts", "elapsed_seconds"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: result[key] for key in columns} for result in results)
        print(f"Result: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
