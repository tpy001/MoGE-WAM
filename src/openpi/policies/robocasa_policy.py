"""Three independent camera inputs and unchanged RoboCasa action semantics."""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.libero_policy import _parse_image

RGB_KEYS = (
    ("observation/left_image", "base_0_rgb"),
    ("observation/right_image", "left_wrist_0_rgb"),
    ("observation/wrist_image", "right_wrist_0_rgb"),
)
FLOW_KEYS = (
    ("observation/left_flow_image", "flow_img"),
    ("observation/right_flow_image", "wrist_flow_img"),
    ("observation/wrist_flow_image", "right_wrist_flow_img"),
)


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (53,):
            raise ValueError(f"Expected RoboCasa state [53], got {state.shape}.")
        inputs = {
            "state": state,
            "image": {slot: _parse_image(data[key]) for key, slot in RGB_KEYS},
            "image_mask": {slot: np.True_ for _, slot in RGB_KEYS},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 12:
                raise ValueError(f"Expected RoboCasa actions [H,12], got {actions.shape}.")
            inputs["actions"] = actions
        present = [key in data for key, _ in FLOW_KEYS]
        if any(present) and not all(present):
            raise ValueError("RoboCasa future flow requires left, right and wrist views together.")
        if all(present):
            for key, slot in FLOW_KEYS:
                inputs[slot] = np.asarray(data[key])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :12]}
