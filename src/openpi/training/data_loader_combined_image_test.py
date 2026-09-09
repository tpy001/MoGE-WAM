from types import SimpleNamespace

import torch

from openpi.training import data_loader


class _Dataset:
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.item


def test_combined_image_dataset_splits_three_by_two_layout():
    num_frames, channels, height, width = 4, 3, 2, 3
    tiles = torch.empty(num_frames, 3, 2, channels, height, width)
    for frame in range(num_frames):
        for row in range(3):
            for camera in range(2):
                tiles[frame, row, camera].fill_(frame * 30 + row * 10 + camera)

    combined = tiles.permute(0, 3, 1, 4, 2, 5).reshape(
        num_frames, channels, 3 * height, 2 * width
    ) / 255.0
    sample = data_loader.CombinedImageDataset(_Dataset({"combined_image": combined}))[0]

    assert sample["image"].shape == (height, width, channels)
    assert sample["wrist_image"].shape == (height, width, channels)
    assert sample["simvla_future_depth_image"].shape == (num_frames, height, width, channels)
    assert sample["simvla_future_flow_wrist_image"].shape == (num_frames, height, width, channels)
    assert sample["image"].dtype == torch.uint8
    assert sample["image"][0, 0, 0].item() == 0
    assert sample["wrist_image"][0, 0, 0].item() == 1
    assert sample["simvla_future_depth_image"][:, 0, 0, 0].tolist() == [10, 40, 70, 100]
    assert sample["simvla_future_flow_wrist_image"][:, 0, 0, 0].tolist() == [21, 51, 81, 111]


def test_combined_image_gets_future_timestamps():
    timestamps = {}
    metadata = SimpleNamespace(fps=20, features={"combined_image": {}})
    model_config = SimpleNamespace(future_frame_offsets=(0, 8, 16, 24))

    data_loader._add_combined_image_timestamps(timestamps, metadata, model_config)  # noqa: SLF001

    assert timestamps == {"combined_image": [0.0, 0.4, 0.8, 1.2]}
