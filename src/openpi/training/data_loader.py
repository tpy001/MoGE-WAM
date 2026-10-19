from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
try:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError as error:
    if error.name != "lerobot.datasets":
        raise
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

# import openpi.models.model as _model
import openpi.models.model_tavla as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.shared.effort_type import EffortType


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")
    
class DelayAwareDataset(Dataset):
    def __init__(
        self,
        dataset: lerobot_dataset.LeRobotDataset,
        max_delay: int,
        prob_zero: float = 0.5,  #  0延时的概率
        alpha: float = 1.0,
        beta: float = 3.0,
        image_keys: list[str] | None = None,
    ):
        self._dataset = dataset
        self._max_delay = max_delay
        self._prob_zero = prob_zero 
        self._dist = torch.distributions.Beta(alpha, beta)
        
        if image_keys is None:
            sample_keys = self._dataset[0].keys()
            self._image_keys = [k for k in sample_keys if "image" in k]
        else:
            self._image_keys = image_keys

    def __getitem__(self, index: int) -> dict:
        current_sample = self._dataset[index]

        # ---------------------------------------------------
        # 1. 先判定是否命中 "绝对 0 延时"
        if torch.rand(1).item() < self._prob_zero:
            delay_frac = 0.0
        else:
            # 2. 如果没命中，则从 Beta 分布采样剩余的延时
            # 注意：Beta 可能会采样出非常接近 0 的数，这没关系，代表"微小延时"
            delay_frac = self._dist.sample().item()
        # ---------------------------------------------------

        delay_steps = int(delay_frac * self._max_delay)
        
        delayed_index = index - delay_steps
        # ... (边界检查) ...
        actual_delay_steps = index - delayed_index
        
        if self._max_delay > 0:
            normalized_delay = actual_delay_steps / self._max_delay
        else:
            normalized_delay = 0.0

        current_sample["observation.delay"] = torch.tensor([normalized_delay], dtype=torch.float32)
        
        if delayed_index == index:
            return current_sample
            
        # 7. 读取旧数据并替换图像
        delayed_sample = self._dataset[delayed_index]

        for key in self._image_keys:
            if key in delayed_sample:
                current_sample[key] = delayed_sample[key]

        return current_sample

    def __len__(self) -> int:
        return len(self._dataset)
    


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _RowFirstLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Avoid materializing a complete HF column for every delta query."""

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            absolute_to_relative_idx = getattr(self, "_absolute_to_relative_idx", None)
            relative_indices = (
                indices
                if absolute_to_relative_idx is None
                else [absolute_to_relative_idx[index] for index in indices]
            )
            # Row-first indexing is important here. ``hf_dataset[key][indices]``
            # runs the HF transform over the entire column before selecting a few
            # delta rows (hundreds of thousands of torch.tensor calls per sample).
            rows = self.hf_dataset[relative_indices]
            result[key] = torch.stack(rows[key])
        return result



class CombinedImageDataset(Dataset):
    """Split a v3 ``combined_image`` video into OpenPI's expected image keys."""

    def __init__(self, dataset: Dataset, *, use_future_rgb_instead_of_flow: bool = False) -> None:
        self.dataset = dataset
        self.use_future_rgb_instead_of_flow = use_future_rgb_instead_of_flow

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_uint8_hwc(value: torch.Tensor) -> torch.Tensor:
        value = value.movedim(-3, -1).contiguous()
        if torch.is_floating_point(value):
            if value.numel() and value.max() <= 2.0:
                value = value * 255.0
            value = value.round().clamp(0, 255).to(torch.uint8)
        return value

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = self.dataset[index]
        combined = torch.as_tensor(item["combined_image"])
        if combined.ndim == 3:
            combined = combined.unsqueeze(0)
        if combined.ndim != 4:
            raise ValueError(f"Expected combined_image [T,C,3H,2W], got {tuple(combined.shape)}.")

        num_frames, channels, combined_height, combined_width = combined.shape
        if combined_height % 3 or combined_width % 2:
            raise ValueError(
                "combined_image spatial shape must be divisible by (3, 2), "
                f"got {(combined_height, combined_width)}."
            )

        tile_height = combined_height // 3
        tile_width = combined_width // 2
        tiles = combined.reshape(
            num_frames,
            channels,
            3,
            tile_height,
            2,
            tile_width,
        ).permute(0, 2, 4, 1, 3, 5)

        rgb = self._to_uint8_hwc(tiles[0, 0])
        depth = self._to_uint8_hwc(tiles[:, 1])
        # Keep the same teacher input slots and model parameters for the RGB ablation.
        teacher_row = 0 if self.use_future_rgb_instead_of_flow else 2
        flow = self._to_uint8_hwc(tiles[:, teacher_row])
        return {
            **item,
            "image": rgb[0],
            "wrist_image": rgb[1],
            "simvla_future_depth_image": depth[:, 0],
            "simvla_future_depth_wrist_image": depth[:, 1],
            "simvla_future_flow_image": flow[:, 0],
            "simvla_future_flow_wrist_image": flow[:, 1],
        }

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


def _add_combined_image_timestamps(
    delta_timestamps: dict[str, list[float]],
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
    model_config: _model.BaseModelConfig,
) -> None:
    """Request the sparse combined-image frames used by the current distillation model."""
    offsets = getattr(model_config, "future_frame_offsets", None)
    if not offsets:
        raise ValueError("The multi-expert model requires non-empty future_frame_offsets.")
    if "combined_image" not in dataset_meta.features:
        raise KeyError(
            "The current multi-expert configuration requires the Dataset v3 "
            f"'combined_image' feature, got {list(dataset_meta.features)}."
        )

    offsets = tuple(int(offset) for offset in offsets)
    timestamps = [offset / dataset_meta.fps for offset in offsets]
    delta_timestamps["combined_image"] = timestamps
    logging.info("Combined image: frame_offsets=%s", offsets)


def _task_index_to_prompt(tasks) -> dict[int, str]:
    """Convert LeRobot v0.4.4 task metadata to OpenPI's task mapping."""
    if hasattr(tasks, "iterrows"):
        return {
            int(row["task_index"]): str(task)
            for task, row in tasks.iterrows()
        }
    return dict(tasks)

def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a LeRobot v0.4.4 / Dataset v3 training dataset.

    Sparse combined-image loading is handled natively by LeRobot through
    delta_timestamps and split into the model's RGB, flow, and depth inputs below.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=data_config.dataset_root,
    )

    # Future action chunk.
    delta_timestamps: dict[str, list[float]] = {
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }

    # Effort history / future effort.
    if "observation.effort" in dataset_meta.features:
        delta_timestamps["observation.effort"] = [
            t / dataset_meta.fps for t in data_config.effort_history
        ]

        if model_config.effort_type in (
            EffortType.EXPERT_FUT,
            EffortType.EXPERT_HIS_C_FUT,
            EffortType.EXPERT_HIS_C_L_FUT,
        ):
            delta_timestamps["observation.effort"] += [
                (t + 1) / dataset_meta.fps for t in range(action_horizon)
            ]

    if "combined_image" in dataset_meta.features:
        _add_combined_image_timestamps(
            delta_timestamps,
            dataset_meta,
            model_config,
        )

    dataset = _RowFirstLeRobotDataset(
        repo_id,
        root=data_config.dataset_root,
        delta_timestamps=delta_timestamps,
        # video_backend=(
        #     "pyav" if lerobot_dataset.__name__.startswith("lerobot.common.") else "torchcodec"
        # ),
        video_backend="torchcodec"
    )

    if "combined_image" in dataset_meta.features:
        use_rgb = bool(getattr(model_config, "use_future_rgb_instead_of_flow", False))
        logging.info("Future teacher input modality: %s", "rgb" if use_rgb else "flow")
        dataset = CombinedImageDataset(dataset, use_future_rgb_instead_of_flow=use_rgb)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(_task_index_to_prompt(dataset_meta.tasks))],
        )

    return dataset

def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            in_order=False,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
