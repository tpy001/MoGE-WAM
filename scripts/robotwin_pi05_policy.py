"""RoboTwin qpos adapter for this repository's pi05_robotwin_rgb policy."""

from collections import deque
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client


class LocalPolicyClient:
    """Allow the first JAX compilation to finish without WebSocket keepalive timeouts."""

    def __init__(self, port):
        self.connection = websockets.sync.client.connect(
            f"ws://127.0.0.1:{port}",
            compression=None,
            max_size=None,
            ping_interval=None,
            # Another client may trigger the initial JAX compilation during our handshake.
            open_timeout=180,
        )
        self.metadata = msgpack_numpy.unpackb(self.connection.recv())
        self.session_id = uuid.uuid4().hex

    def infer(self, observation):
        self.connection.send(msgpack_numpy.packb(dict(observation, _eval_session=self.session_id)))
        response = self.connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server inference failed:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self):
        pass


def encode_observation(observation, instruction):
    cameras = observation["observation"]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Expected a finite 14-D joint state, got {state.shape}")
    return {
        "observation/image": cameras["head_camera"]["rgb"],
        "observation/left_wrist_image": cameras["left_camera"]["rgb"],
        "observation/right_wrist_image": cameras["right_camera"]["rgb"],
        "observation/state": state,
        "prompt": instruction,
    }


class Pi05RoboTwinPolicy:
    def __init__(self, client, replan_steps=24, trace_path=None):
        if not 1 <= replan_steps <= 32:
            raise ValueError("replan_steps must be between 1 and 32")
        self.client = client
        self.replan_steps = replan_steps
        self.pending = deque()
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.previous_action = None
        self.previous_next = None
        self.episode_index = 0
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.inference_calls = 0
        self.inference_seconds = 0.0
        self.executed_actions = 0
        self.action_digest = hashlib.sha256()

    def should_request_observation(self):
        return not self.pending

    def step(self, env, observation):
        if not self.pending:
            inputs = encode_observation(observation, env.get_instruction())
            started = time.perf_counter()
            actions = np.asarray(self.client.infer(inputs)["actions"], dtype=np.float32)
            self.inference_seconds += time.perf_counter() - started
            if actions.shape != (32, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite absolute actions [32,14], got {actions.shape}")
            if self.trace_path is not None:
                record = {
                    "episode": self.episode_index,
                    "chunk": self.inference_calls,
                    "executed_actions": self.executed_actions,
                    "state": inputs["observation/state"].tolist(),
                    "real_state": list(env.robot.get_left_arm_real_jointState())
                    + list(env.robot.get_right_arm_real_jointState()),
                    "first_action": actions[0].tolist(),
                    "previous_action": None if self.previous_action is None else self.previous_action.tolist(),
                    "previous_next": None if self.previous_next is None else self.previous_next.tolist(),
                }
                with self.trace_path.open("a") as stream:
                    stream.write(json.dumps(record, default=float) + "\n")
                self.previous_next = actions[self.replan_steps].copy() if self.replan_steps < len(actions) else None
            self.pending.extend(actions[: self.replan_steps])
            self.inference_calls += 1
        action = self.pending.popleft()
        self.action_digest.update(action.tobytes())
        self.executed_actions += 1
        env.take_action(action, action_type="qpos")
        if self.trace_path is not None:
            self.previous_action = action.copy()

    def reset(self):
        self.pending.clear()
        self.previous_action = None
        self.previous_next = None
        self.episode_index += 1
        self.client.reset()


def get_model(args):
    client = LocalPolicyClient(args["port"])
    metadata = client.metadata
    if metadata.get("checkpoint") != args["checkpoint"] or metadata.get("config") != args.get("config", "pi05_robotwin_rgb"):
        raise ValueError(f"Connected to the wrong policy server: {metadata}")
    return Pi05RoboTwinPolicy(
        client, args["replan_steps"], trace_path=Path(args["output_dir"]) / "chunk_boundaries.jsonl"
    )


def eval(task_env, model, observation):
    model.step(task_env, observation)


def reset_model(model):
    model.reset()


class SessionPolicy:
    """Give each simulator an independent RNG stream while sharing model weights."""

    def __init__(self, policy, make_rng):
        self.policy = policy
        self.make_rng = make_rng
        self.rngs = {}

    def infer(self, observation):
        observation = dict(observation)
        session = observation.pop("_eval_session", "default")
        # WebsocketPolicyServer runs infer synchronously, with no await in this section.
        self.policy._rng = self.rngs.get(session, None)  # noqa: SLF001
        if self.policy._rng is None:  # noqa: SLF001
            self.policy._rng = self.make_rng()  # noqa: SLF001
        result = self.policy.infer(observation)
        self.rngs[session] = self.policy._rng  # noqa: SLF001
        return result
