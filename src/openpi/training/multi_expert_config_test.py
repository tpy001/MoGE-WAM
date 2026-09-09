from pathlib import Path

from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders


def test_pretrain_uses_libero40_and_20k_cosine_schedule():
    train_config = config.get_config("pi05_latent_flow_depth_simvla_pretrain_libero40")
    assert train_config.data.repo_id == "physical-intelligence/libero_40_lerobot_rgb_flow_depth_mp4"
    assert train_config.num_train_steps == 20_000
    assert train_config.model.action_only_pretrain
    assert isinstance(train_config.lr_schedule, optimizer.CosineDecaySchedule)
    assert train_config.lr_schedule.warmup_steps == 1_000
    assert train_config.lr_schedule.peak_lr == 2.5e-5
    assert train_config.lr_schedule.decay_steps == 20_000
    assert train_config.lr_schedule.decay_lr == 2.5e-6


def test_posttrain_uses_libero40_and_requires_checkpoint_override():
    train_config = config.get_config("pi05_latent_flow_depth_simvla_posttrain_libero40")
    assert train_config.data.repo_id == "physical-intelligence/libero_40_lerobot_rgb_flow_depth_mp4"
    assert train_config.num_train_steps == 30_000
    assert not train_config.model.action_only_pretrain
    assert train_config.model.future_flow_contrast_loss_weight == 0.1
    assert train_config.model.future_depth_contrast_loss_weight == 0.1
    assert isinstance(train_config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert train_config.weight_loader.params_path == ""


def test_training_does_not_materialize_sharded_images_for_wandb():
    source = (Path(__file__).parents[3] / "scripts" / "train.py").read_text()
    assert "np.array(img[i])" not in source
    assert 'wandb.log({"camera_views"' not in source
