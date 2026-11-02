"""Decode the RoboCasa RGB/flow mosaic without changing episode sampling."""
import json
from pathlib import Path

import torch

CAMERA_NAMES = ("left_view", "right_view", "wrist_view")


class RoboCasaMosaicDataset:
    def __init__(self, dataset, layout_path, task_descriptions):
        self.dataset = dataset
        self.layout = json.loads(Path(layout_path).read_text())
        self.tiles = {tile["source_key"]: tile for tile in self.layout["tiles"]}
        self.task_descriptions = task_descriptions

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_uint8_hwc(value):
        value = value.movedim(-3, -1).contiguous()
        if torch.is_floating_point(value):
            if value.numel() and value.max() <= 2.0:
                value = value * 255.0
            value = value.round().clamp(0, 255).to(torch.uint8)
        return value

    def __getitem__(self, index):
        item = self.dataset[index]
        mosaic = torch.as_tensor(item[self.layout["feature"]])
        expected = (4, 3, self.layout["height"], self.layout["width"])
        if tuple(mosaic.shape) != expected:
            raise ValueError(f"Expected RoboCasa mosaic {expected}, got {tuple(mosaic.shape)}.")
        result = dict(item)
        for name in CAMERA_NAMES:
            for suffix in ("", "_flow"):
                tile = self.tiles[f"observation.images.{name}{suffix}"]
                crop = mosaic[..., slice(*tile["y_slice"]), slice(*tile["x_slice"])]
                result[f"robocasa_{name}{suffix}"] = self._to_uint8_hwc(crop if suffix else crop[0])
        description_index = int(item["annotation.human.action.task_description"].item())
        result["prompt"] = self.task_descriptions[description_index]
        return result

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)
