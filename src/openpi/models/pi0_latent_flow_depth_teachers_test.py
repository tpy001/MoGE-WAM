import inspect

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_latent_flow_depth_teachers as teachers


def test_spatiotemporal_encoder_matches_simvla_shape():
    encoder = teachers.SpatiotemporalEncoder3D(rngs=nnx.Rngs(0))
    output = jax.eval_shape(
        encoder,
        jax.ShapeDtypeStruct((1, 2, 4, 256, 256, 3), jnp.float32),
    )
    assert output.shape == (1, 2, 2, 16, 16, 512)


def test_future_encoder_uses_two_layer_eight_head_qformer():
    encoder = teachers.FutureModalityEncoder(num_queries=16, rngs=nnx.Rngs(0))
    output = jax.eval_shape(
        encoder,
        jax.ShapeDtypeStruct((1, 2, 4, 256, 256, 3), jnp.float32),
    )
    assert output.shape == (1, 16, 512)
    assert encoder.qformer.layers["layer_0"].num_heads == 8
    assert encoder.qformer.layers["layer_1"].num_heads == 8


def test_flow_and_depth_future_encoders_do_not_share_parameters():
    rngs = nnx.Rngs(0)
    flow = teachers.FutureModalityEncoder(num_queries=16, rngs=rngs)
    depth = teachers.FutureModalityEncoder(num_queries=16, rngs=rngs)
    assert flow.backbone is not depth.backbone
    assert flow.qformer is not depth.qformer
    assert not np.shares_memory(
        np.asarray(flow.qformer.query_tokens.value),
        np.asarray(depth.qformer.query_tokens.value),
    )


def test_future_encoder_rejects_noncanonical_volume_shape():
    encoder = teachers.SpatiotemporalEncoder3D(rngs=nnx.Rngs(0))
    with np.testing.assert_raises_regex(ValueError, "Expected future RGB volume"):
        encoder(jnp.zeros((1, 2, 3, 256, 256, 3), dtype=jnp.float32))


def test_future_encoder_parameter_paths_are_string_only():
    encoder = teachers.FutureModalityEncoder(num_queries=16, rngs=nnx.Rngs(0))
    params = nnx.state(encoder).to_pure_dict()
    flattened = flax.traverse_util.flatten_dict(params, sep="/")
    assert flattened
    assert all(isinstance(path, str) for path in flattened)


def test_pretrain_uses_equal_weight_action_loss():
    source = inspect.getsource(teachers.Pi0LatentFlowDepthTeachers.compute_loss_with_stats)
    assert "(student_action_loss + flow_teacher_action_loss + depth_teacher_action_loss) / 3.0" in source


def test_posttrain_stops_teacher_gradients_and_removes_cosine_alignment():
    source = inspect.getsource(teachers.Pi0LatentFlowDepthTeachers.compute_loss_with_stats)
    assert source.count("jax.lax.stop_gradient") == 2
    assert "_cosine_distance_masked" not in source
