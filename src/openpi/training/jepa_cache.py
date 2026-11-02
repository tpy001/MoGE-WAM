"""Optional, row-aligned access to precomputed JEPA episode shards."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch


JEPA_KEYS = ("jepa_features", "jepa_valid_token_mask", "jepa_future_is_pad")


class JepaCachedDataset:
    def __init__(self, dataset, cache_dir: str | None, *, dataset_root, future_offset: int = 10):
        if not cache_dir:
            raise ValueError("jepa_cache_dir is required when jepa_enabled=True.")
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if manifest["shape"] != [144, 1024] or manifest["dtype"] != "float16" or manifest["frames"] != [0, future_offset]:
            raise ValueError(f"Expected FP16 JEPA cache [144,1024] for frames [0,{future_offset}].")
        if Path(manifest["dataset"]).resolve() != Path(dataset_root).resolve():
            raise ValueError("JEPA cache source dataset does not match the training dataset root.")
        self._shards = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = OrderedDict()
        state["_pid"] = None
        return state

    def _episode(self, episode_index):
        if self._pid != os.getpid():
            self._shards.clear()
            self._pid = os.getpid()
        if episode_index not in self._shards:
            path = self.cache_dir / "episodes" / f"episode_{episode_index:06d}.pt"
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if (shard["episode_index"] != episode_index or shard["features"].shape[1:] != (144, 1024)
                    or shard["features"].dtype != torch.float16 or shard["valid_token_mask"].shape != (144,)):
                raise ValueError(f"Invalid JEPA episode shard: {path}")
            self._shards[episode_index] = shard
            if len(self._shards) > 32:
                self._shards.popitem(last=False)
        self._shards.move_to_end(episode_index)
        return self._shards[episode_index]

    def __getitem__(self, index):
        item = self.dataset[index]
        episode_index, frame_index = int(item["episode_index"]), int(item["frame_index"])
        shard = self._episode(episode_index)
        if not 0 <= frame_index < len(shard["features"]):
            raise ValueError(f"JEPA frame out of range: episode={episode_index}, frame={frame_index}")
        if int(shard["index"][frame_index]) != int(item["index"]):
            raise ValueError(f"JEPA global row mismatch: episode={episode_index}, frame={frame_index}")
        return {
            **item,
            "jepa_features": shard["features"][frame_index].numpy().copy(),
            "jepa_valid_token_mask": shard["valid_token_mask"].numpy().copy(),
            "jepa_future_is_pad": np.bool_(shard["future_is_pad"][frame_index]),
        }
