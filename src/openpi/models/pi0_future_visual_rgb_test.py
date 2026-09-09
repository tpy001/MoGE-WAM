import numpy as np

from openpi import transforms
from openpi.models import model_tavla
from openpi.models import pi0_config
from openpi.policies import libero_policy


def test_future_visual_rgb_config_uses_combined_image_offsets():
    config = pi0_config.Pi0LatentFlowDepthTeachersConfig()
    assert config.future_frame_offsets == (0, 8, 16, 24)


def test_flow_depth_transform_preserves_uint8_until_observation_conversion():
    frame = np.full((16, 20, 3), 255, dtype=np.uint8)
    volume = np.stack([frame] * 5)
    transformed = libero_policy.LiberoFlowDepthInputs(model_type=model_tavla.ModelType.PI05)(
        {
            "observation/image": frame,
            "observation/wrist_image": frame,
            "observation/flow_image": volume,
            "observation/flow_wrist_image": volume,
            "observation/depth_image": volume,
            "observation/depth_wrist_image": volume,
            "observation/state": np.zeros(8, dtype=np.float32),
        }
    )
    assert transformed["flow_img"].dtype == np.uint8
    transformed["image"] = {key: value[None] for key, value in transformed["image"].items()}
    transformed["image_mask"] = {
        key: np.asarray([value], dtype=np.bool_) for key, value in transformed["image_mask"].items()
    }
    transformed["state"] = transformed["state"][None]
    for key in ("flow_img", "wrist_flow_img", "depth_img", "wrist_depth_img"):
        transformed[key] = transformed[key][None]
    observation = model_tavla.Observation.from_dict(transformed)
    assert observation.flow_img.shape == (1, 5, 16, 20, 3)
    np.testing.assert_array_equal(observation.flow_img, 1.0)
    np.testing.assert_array_equal(observation.depth_img, 1.0)


def test_future_rgb_volumes_are_resized_to_256_without_touching_current_images():
    current = np.zeros((20, 24, 3), dtype=np.uint8)
    volume = np.zeros((4, 16, 20, 3), dtype=np.uint8)
    data = {
        "image": {"base_0_rgb": current},
        "flow_img": volume,
        "wrist_flow_img": volume,
        "depth_img": volume,
        "wrist_depth_img": volume,
    }
    result = transforms.ResizeFutureImageVolumes()(data)
    assert result["image"]["base_0_rgb"].shape == (20, 24, 3)
    for key in ("flow_img", "wrist_flow_img", "depth_img", "wrist_depth_img"):
        assert result[key].shape == (4, 256, 256, 3)
        assert result[key].dtype == np.uint8
