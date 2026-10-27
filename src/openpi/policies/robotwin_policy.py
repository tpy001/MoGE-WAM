"""RoboTwin two-view RGB inputs shared by training and policy inference."""

import dataclasses

import cv2
import numpy as np

from openpi import transforms

RGB_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
FLOW_VIDEO_KEY = "observation.images.cam_high_flow"


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB frame, got shape {image.shape}")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    load_flow: bool = False
    allow_missing_flow: bool = False

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["observation/image"])
        left = _parse_image(data["observation/left_wrist_image"])
        right = _parse_image(data["observation/right_wrist_image"])
        wrists = np.concatenate(
            [
                cv2.resize(left, (640, 320), interpolation=cv2.INTER_AREA),
                cv2.resize(right, (640, 320), interpolation=cv2.INTER_AREA),
            ],
            axis=0,
        )
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-dimensional RoboTwin state, got {state.shape}")
        inputs = {
            "state": state,
            "image": {"base_0_rgb": head, "left_wrist_0_rgb": wrists},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected 14-dimensional RoboTwin actions, got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if self.load_flow:
            value = data.get("observation/flow_image")
            if value is None:
                if not self.allow_missing_flow:
                    raise KeyError("observation/flow_image")
            else:
                value = np.asarray(value)
                inputs["flow_img"] = (
                    np.stack([_parse_image(frame) for frame in value])
                    if value.ndim == 4 else _parse_image(value)
                )
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinAppearanceAugmentation(transforms.DataTransformFn):
    """Training-only C2R appearance changes on raw cameras before composition."""

    def __call__(self, data: dict) -> dict:
        import torch
        from openpi.policies.robotwin_appearance import AppearanceRandomizer

        keys = ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
        views = [torch.from_numpy(_parse_image(data[key]).copy()).permute(2, 0, 1).float()[None] / 255
                 for key in keys]
        views = AppearanceRandomizer(
            p=0.8, style={"enabled": True}, fourier={"enabled": True},
        )(views)
        result = dict(data)
        for key, view in zip(keys, views, strict=True):
            result[key] = (view[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        return result


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :14]}
