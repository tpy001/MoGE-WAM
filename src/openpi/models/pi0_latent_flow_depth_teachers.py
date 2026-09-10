import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.models.pi0_tavla import make_attn_mask
from openpi.models.pi0_tavla import posemb_sincos
from openpi.shared import array_typing as at

_FUTURE_HIDDEN_DIM = 512
_FUTURE_NUM_VIEWS = 2
_FUTURE_NUM_FRAMES = 4
_FUTURE_OUTPUT_TIME = 2
_FUTURE_OUTPUT_GRID = 16
_QFORMER_LAYERS = 2
_QFORMER_HEADS = 8
_QFORMER_MLP_DIM = 2048


class ResBlock3D(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int], *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            strides=stride,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(out_channels, num_groups=32, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(out_channels, num_groups=32, rngs=rngs)
        self.skip = (
            None
            if in_channels == out_channels and stride == (1, 1, 1)
            else nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                strides=stride,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
        )

    def __call__(self, x):
        residual = x if self.skip is None else self.skip(x)
        x = nnx.swish(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return nnx.swish(x + residual)


class SpatiotemporalEncoder3D(nnx.Module):
    """Encode a fixed [B,V,4,256,256,3] future RGB volume."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.stem = nnx.Conv(
            3,
            64,
            kernel_size=(3, 7, 7),
            strides=(1, 2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.stem_norm = nnx.GroupNorm(64, num_groups=32, rngs=rngs)
        self.stages = nnx.Dict(
            stage_0=ResBlock3D(64, 128, (1, 2, 2), rngs=rngs),
            stage_1=ResBlock3D(128, 256, (2, 2, 2), rngs=rngs),
            stage_2=ResBlock3D(256, 384, (1, 2, 2), rngs=rngs),
            stage_3=ResBlock3D(384, 512, (1, 1, 1), rngs=rngs),
        )

    def __call__(self, volume):
        x = jnp.asarray(volume, dtype=jnp.float32)
        if x.ndim != 6 or x.shape[1:] != (
            _FUTURE_NUM_VIEWS,
            _FUTURE_NUM_FRAMES,
            256,
            256,
            3,
        ):
            raise ValueError(f"Expected future RGB volume [B,2,4,256,256,3], got {x.shape}.")
        batch_size, views, time, height, width, channels = x.shape
        x = x.reshape(batch_size * views, time, height, width, channels)
        x = nnx.swish(self.stem_norm(self.stem(x)))
        for stage_index in range(4):
            x = self.stages[f"stage_{stage_index}"](x)
        _, time, height, width, channels = x.shape
        x = x.reshape(batch_size, views, time, height, width, channels)
        expected = (_FUTURE_NUM_VIEWS, _FUTURE_OUTPUT_TIME, _FUTURE_OUTPUT_GRID, _FUTURE_OUTPUT_GRID, 512)
        if x.shape[1:] != expected:
            raise ValueError(f"Unexpected 3D encoder output {x.shape}; expected [B,{','.join(map(str, expected))}].")
        return x


class QFormerLayer(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.num_heads = _QFORMER_HEADS
        self.query_norm = nnx.LayerNorm(_FUTURE_HIDDEN_DIM, rngs=rngs)
        self.context_norm = nnx.LayerNorm(_FUTURE_HIDDEN_DIM, rngs=rngs)
        self.q_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, _FUTURE_HIDDEN_DIM, rngs=rngs)
        self.k_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, _FUTURE_HIDDEN_DIM, rngs=rngs)
        self.v_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, _FUTURE_HIDDEN_DIM, rngs=rngs)
        self.out_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, _FUTURE_HIDDEN_DIM, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(_FUTURE_HIDDEN_DIM, rngs=rngs)
        self.mlp_in = nnx.Linear(_FUTURE_HIDDEN_DIM, _QFORMER_MLP_DIM, rngs=rngs)
        self.mlp_out = nnx.Linear(_QFORMER_MLP_DIM, _FUTURE_HIDDEN_DIM, rngs=rngs)

    def __call__(self, query, context):
        head_dim = _FUTURE_HIDDEN_DIM // _QFORMER_HEADS
        q = self.q_proj(self.query_norm(query)).reshape(query.shape[0], query.shape[1], _QFORMER_HEADS, head_dim)
        context = self.context_norm(context)
        keys = self.k_proj(context).reshape(context.shape[0], context.shape[1], _QFORMER_HEADS, head_dim)
        values = self.v_proj(context).reshape(context.shape[0], context.shape[1], _QFORMER_HEADS, head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, keys) / jnp.sqrt(jnp.asarray(head_dim, dtype=q.dtype))
        attention = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(values.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", attention, values).reshape(query.shape)
        query = query + self.out_proj(attended)
        mlp = nnx.swish(self.mlp_in(self.mlp_norm(query)))
        return query + self.mlp_out(mlp)


class QFormer(nnx.Module):
    def __init__(self, num_queries: int, *, rngs: nnx.Rngs):
        self.query_tokens = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (num_queries, _FUTURE_HIDDEN_DIM), dtype=jnp.float32)
        )
        self.layers = nnx.Dict(**{f"layer_{index}": QFormerLayer(rngs=rngs) for index in range(_QFORMER_LAYERS)})
        self.output_norm = nnx.LayerNorm(_FUTURE_HIDDEN_DIM, rngs=rngs)

    def __call__(self, context):
        query = jnp.asarray(self.query_tokens.value, dtype=context.dtype)
        query = jnp.broadcast_to(query[None], (context.shape[0], *query.shape))
        for layer_index in range(_QFORMER_LAYERS):
            query = self.layers[f"layer_{layer_index}"](query, context)
        return self.output_norm(query)


class FutureModalityEncoder(nnx.Module):
    def __init__(self, num_queries: int, *, rngs: nnx.Rngs):
        self.backbone = SpatiotemporalEncoder3D(rngs=rngs)
        self.temporal_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, 1, _FUTURE_OUTPUT_TIME, 1, 1, 512), dtype=jnp.float32)
        )
        self.spatial_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, 1, 1, _FUTURE_OUTPUT_GRID, _FUTURE_OUTPUT_GRID, 512),
                dtype=jnp.float32,
            )
        )
        self.view_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, _FUTURE_NUM_VIEWS, 1, 1, 1, 512), dtype=jnp.float32)
        )
        self.qformer = QFormer(num_queries, rngs=rngs)

    def __call__(self, volume):
        context = self.backbone(volume)
        context = (
            context
            + jnp.asarray(self.temporal_embedding.value, dtype=context.dtype)
            + jnp.asarray(self.spatial_embedding.value, dtype=context.dtype)
            + jnp.asarray(self.view_embedding.value, dtype=context.dtype)
        )
        context = context.reshape(context.shape[0], -1, context.shape[-1])
        return self.qformer(context)


class Pi0LatentFlowDepthTeachers(_model.BaseModel):
    """Single student expert distilled from separate flow and depth teacher experts."""

    def __init__(self, config: pi0_config.Pi0LatentFlowDepthTeachersConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.effort_type = config.effort_type
        self.distill_layer_indices = tuple(int(i) for i in config.distill_layer_indices)
        self.student_action_loss_weight = float(config.student_action_loss_weight)
        self.flow_teacher_action_loss_weight = float(config.flow_teacher_action_loss_weight)
        self.depth_teacher_action_loss_weight = float(config.depth_teacher_action_loss_weight)
        self.future_flow_contrast_loss_weight = float(config.future_flow_contrast_loss_weight)
        self.future_depth_contrast_loss_weight = float(config.future_depth_contrast_loss_weight)
        self.distill_loss_type = config.distill_loss_type
        self.distill_contrast_temperature = float(config.distill_contrast_temperature)
        self.flow_token_count = int(config.flow_token_count)
        self.depth_token_count = int(config.depth_token_count)
        self.future_frame_offsets = tuple(int(offset) for offset in config.future_frame_offsets)
        self.action_only_pretrain = bool(config.action_only_pretrain)
        self.freeze_future_encoders = bool(config.freeze_future_encoders)
        self.student_future_query_noise_scale_max = float(config.student_future_query_noise_scale_max)
        self.student_future_query_noise_start_ratio = float(config.student_future_query_noise_start_ratio)
        self.student_future_query_noise_end_ratio = float(config.student_future_query_noise_end_ratio)
        self.uses_train_progress = True

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        student_config = _gemma.get_config(config.action_expert_variant)
        flow_teacher_config = _gemma.get_config(config.flow_teacher_expert_variant)
        depth_teacher_config = _gemma.get_config(config.depth_teacher_expert_variant)
        self.student_width = int(student_config.width)
        self.flow_teacher_width = int(flow_teacher_config.width)
        self.depth_teacher_width = int(depth_teacher_config.width)
        self.flow_projector_hidden_dim = int(
            config.distill_projector_hidden_dim
            if config.distill_projector_hidden_dim is not None
            else flow_teacher_config.width
        )
        self.depth_projector_hidden_dim = int(
            config.depth_distill_projector_hidden_dim
            if config.depth_distill_projector_hidden_dim is not None
            else depth_teacher_config.width
        )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, student_config, flow_teacher_config, depth_teacher_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True, True, True] if config.pi05 else [False, False, False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self._init_action_path("student", student_config.width, rngs)
        self._init_action_path("flow_teacher", flow_teacher_config.width, rngs)
        self._init_action_path("depth_teacher", depth_teacher_config.width, rngs)

        self.student_future_flow_query = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.flow_token_count, self.student_width), dtype=jnp.float32)
        )
        self.student_future_depth_query = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.depth_token_count, self.student_width), dtype=jnp.float32)
        )

        self.flow_future_encoder = FutureModalityEncoder(self.flow_token_count, rngs=rngs)
        self.depth_future_encoder = FutureModalityEncoder(self.depth_token_count, rngs=rngs)
        self.flow_teacher_token_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, self.flow_teacher_width, rngs=rngs)
        self.depth_teacher_token_proj = nnx.Linear(_FUTURE_HIDDEN_DIM, self.depth_teacher_width, rngs=rngs)

        for layer_ordinal, _ in enumerate(() if self.action_only_pretrain else self.distill_layer_indices):
            setattr(
                self,
                f"flow_distill_proj_in_{layer_ordinal}",
                nnx.Linear(self.student_width, self.flow_projector_hidden_dim, rngs=rngs),
            )
            setattr(
                self,
                f"flow_distill_proj_out_{layer_ordinal}",
                nnx.Linear(self.flow_projector_hidden_dim, self.flow_teacher_width, rngs=rngs),
            )
            setattr(
                self,
                f"depth_distill_proj_in_{layer_ordinal}",
                nnx.Linear(self.student_width, self.depth_projector_hidden_dim, rngs=rngs),
            )
            setattr(
                self,
                f"depth_distill_proj_out_{layer_ordinal}",
                nnx.Linear(self.depth_projector_hidden_dim, self.depth_teacher_width, rngs=rngs),
            )

    def _init_action_path(self, name: str, width: int, rngs: nnx.Rngs) -> None:
        setattr(self, f"state_proj_{name}", nnx.Linear(self.action_dim, width, rngs=rngs))
        setattr(self, f"action_in_proj_{name}", nnx.Linear(self.action_dim, width, rngs=rngs))
        setattr(self, f"action_out_proj_{name}", nnx.Linear(width, self.action_dim, rngs=rngs))
        if self.pi05:
            setattr(self, f"{name}_time_mlp_in", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_time_mlp_out", nnx.Linear(width, width, rngs=rngs))
        else:
            setattr(self, f"{name}_time_mlp_in", nnx.Linear(2 * width, width, rngs=rngs))
            setattr(self, f"{name}_time_mlp_out", nnx.Linear(width, width, rngs=rngs))

    def _student_query_noise_scale(self, train_progress: at.Float[at.Array, ""] | float | None):
        if train_progress is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.clip(jnp.asarray(train_progress, dtype=jnp.float32), 0.0, 1.0)
        start = jnp.asarray(self.student_future_query_noise_start_ratio, dtype=jnp.float32)
        end = jnp.asarray(self.student_future_query_noise_end_ratio, dtype=jnp.float32)
        max_scale = jnp.asarray(self.student_future_query_noise_scale_max, dtype=jnp.float32)
        ramp = (progress - start) / jnp.maximum(end - start, 1e-6)
        return max_scale * jnp.clip(ramp, 0.0, 1.0)

    def _student_query_tokens(self, batch_size, dtype, *, train, noise_rng, train_progress, query_noise_scale=None):
        flow = jnp.asarray(self.student_future_flow_query.value, dtype=dtype)
        depth = jnp.asarray(self.student_future_depth_query.value, dtype=dtype)
        flow = jnp.broadcast_to(flow[None], (batch_size, *flow.shape))
        depth = jnp.broadcast_to(depth[None], (batch_size, *depth.shape))
        noise_scale_f32 = (
            self._student_query_noise_scale(train_progress)
            if query_noise_scale is None
            else jnp.asarray(query_noise_scale, dtype=jnp.float32)
        )
        noise_scale_f32 = jnp.maximum(noise_scale_f32, 0.0)
        if (not train and query_noise_scale is None) or (
            query_noise_scale is None and float(self.student_future_query_noise_scale_max) <= 0.0
        ):
            return flow, depth, jnp.zeros((batch_size,), dtype=jnp.float32)
        if noise_rng is None:
            raise ValueError("noise_rng is required when student future query noise is enabled.")
        flow_rng, depth_rng = jax.random.split(noise_rng)
        noise_scale = noise_scale_f32.astype(dtype)
        flow_rms = jnp.sqrt(jnp.mean(jnp.square(flow.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        depth_rms = jnp.sqrt(jnp.mean(jnp.square(depth.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        flow = flow + noise_scale * flow_rms.astype(dtype) * jax.random.normal(flow_rng, flow.shape, dtype=dtype)
        depth = depth + noise_scale * depth_rms.astype(dtype) * jax.random.normal(depth_rng, depth.shape, dtype=dtype)
        rate = jnp.ones((batch_size,), dtype=jnp.float32) * jnp.where(noise_scale_f32 > 0.0, 1.0, 0.0)
        return flow, depth, rate

    @staticmethod
    def _require_image(image, field_name: str):
        if image is None:
            raise ValueError(f"Pi0LatentFlowDepthTeachers requires `{field_name}`.")
        return image

    def _visual_images(self, obs: _model.Observation, name: str):
        if name == "flow":
            return (
                self._require_image(obs.flow_img, "observation.flow_img"),
                self._require_image(obs.wrist_flow_img, "observation.wrist_flow_img"),
            )
        if name == "depth":
            return (
                self._require_image(obs.depth_img, "observation.depth_img"),
                self._require_image(obs.wrist_depth_img, "observation.wrist_depth_img"),
            )
        raise ValueError(f"Unknown visual teacher: {name}")

    def _compress_visuals(self, obs: _model.Observation, name: str):
        volume = jnp.stack(self._visual_images(obs, name), axis=1)
        tokens = getattr(self, f"{name}_future_encoder")(volume)
        if self.freeze_future_encoders:
            tokens = jax.lax.stop_gradient(tokens)
        return getattr(self, f"{name}_teacher_token_proj")(tokens)

    def embed_prefix(self, obs: _model.Observation):
        tokens = []
        input_mask = []
        ar_mask = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        if obs.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)

    def _embed_action_tokens(self, noisy_actions, timestep, name: str):
        action_tokens = getattr(self, f"action_in_proj_{name}")(noisy_actions)
        width = getattr(self, f"action_in_proj_{name}").out_features
        time_mlp_in = getattr(self, f"{name}_time_mlp_in")
        time_mlp_out = getattr(self, f"{name}_time_mlp_out")
        time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            return action_tokens, time_emb
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = time_mlp_out(action_time_tokens)
        return action_time_tokens, None

    def embed_student_suffix(self, obs, noisy_actions, timestep, *, train, noise_rng, train_progress, query_noise_scale=None):
        flow_query, depth_query, noised_rate = self._student_query_tokens(
            obs.state.shape[0],
            obs.state.dtype,
            train=train,
            noise_rng=noise_rng,
            train_progress=train_progress,
            query_noise_scale=query_noise_scale,
        )
        state = self.state_proj_student(obs.state)[:, None, :]
        actions, adarms = self._embed_action_tokens(noisy_actions, timestep, "student")
        tokens = jnp.concatenate([state, flow_query, depth_query, actions], axis=1)
        ar_mask = jnp.array(
            [False]
            + ([False] * self.flow_token_count)
            + ([False] * self.depth_token_count)
            + [True]
            + ([False] * (actions.shape[1] - 1))
        )
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_), ar_mask, adarms, noised_rate

    def embed_teacher_suffix(self, obs, noisy_actions, timestep, name: str):
        teacher = f"{name}_teacher"
        token_count = self.flow_token_count if name == "flow" else self.depth_token_count
        state = getattr(self, f"state_proj_{teacher}")(obs.state)[:, None, :]
        visual = self._compress_visuals(obs, name)
        actions, adarms = self._embed_action_tokens(noisy_actions, timestep, teacher)
        tokens = jnp.concatenate([state, visual, actions], axis=1)
        ar_mask = jnp.array([False] + ([False] * token_count) + [True] + ([False] * (actions.shape[1] - 1)))
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_), ar_mask, adarms

    def _forward_all_streams(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        student_tokens,
        student_mask,
        student_ar_mask,
        student_adarms,
        flow_tokens,
        flow_mask,
        flow_ar_mask,
        flow_adarms,
        depth_tokens,
        depth_mask,
        depth_ar_mask,
        depth_adarms,
    ):
        bsz = prefix_mask.shape[0]
        p_len = prefix_mask.shape[1]
        student_len = student_mask.shape[1]
        flow_len = flow_mask.shape[1]
        depth_len = depth_mask.shape[1]

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        student_attn = make_attn_mask(student_mask, student_ar_mask)
        flow_attn = make_attn_mask(flow_mask, flow_ar_mask)
        depth_attn = make_attn_mask(depth_mask, depth_ar_mask)

        def suffix_to_prefix(suffix_mask, allow_prefix):
            attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_mask.shape[1])
            attn = jnp.logical_and(attn, suffix_mask[:, :, None])
            return jnp.logical_and(attn, allow_prefix[None, :, None])

        student_prefix = suffix_to_prefix(
            student_mask,
            jnp.array(
                [True]
                + ([True] * self.flow_token_count)
                + ([True] * self.depth_token_count)
                + ([True] * self.action_horizon)
            ),
        )
        flow_prefix = suffix_to_prefix(
            flow_mask,
            jnp.array([True] + ([True] * self.flow_token_count) + ([True] * self.action_horizon)),
        )
        depth_prefix = suffix_to_prefix(
            depth_mask,
            jnp.array([True] + ([True] * self.depth_token_count) + ([True] * self.action_horizon)),
        )

        prefix_row = jnp.concatenate(
            [
                prefix_attn,
                jnp.zeros((bsz, p_len, student_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, flow_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        student_row = jnp.concatenate(
            [
                student_prefix,
                student_attn,
                jnp.zeros((bsz, student_len, flow_len), dtype=jnp.bool_),
                jnp.zeros((bsz, student_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        flow_row = jnp.concatenate(
            [
                flow_prefix,
                jnp.zeros((bsz, flow_len, student_len), dtype=jnp.bool_),
                flow_attn,
                jnp.zeros((bsz, flow_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        depth_row = jnp.concatenate(
            [
                depth_prefix,
                jnp.zeros((bsz, depth_len, student_len), dtype=jnp.bool_),
                jnp.zeros((bsz, depth_len, flow_len), dtype=jnp.bool_),
                depth_attn,
            ],
            axis=-1,
        )
        full_attn = jnp.concatenate([prefix_row, student_row, flow_row, depth_row], axis=1)

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
        student_positions = prefix_len + jnp.cumsum(student_mask, axis=-1) - 1
        flow_positions = prefix_len + jnp.cumsum(flow_mask, axis=-1) - 1
        depth_positions = prefix_len + jnp.cumsum(depth_mask, axis=-1) - 1
        positions = jnp.concatenate([prefix_positions, student_positions, flow_positions, depth_positions], axis=1)
        if self.action_only_pretrain:
            outputs, _ = self.PaliGemma.llm(
                [prefix_tokens, student_tokens, flow_tokens, depth_tokens],
                mask=full_attn,
                positions=positions,
                adarms_cond=[None, student_adarms, flow_adarms, depth_adarms],
            )
            return outputs, ()
        (outputs, layers), _ = self.PaliGemma.llm(
            [prefix_tokens, student_tokens, flow_tokens, depth_tokens],
            mask=full_attn,
            positions=positions,
            adarms_cond=[None, student_adarms, flow_adarms, depth_adarms],
            return_layer_indices=self.distill_layer_indices,
        )
        return outputs, layers

    @staticmethod
    def _tokenwise_contrastive_cosine_loss(student, teacher, token_mask, temperature: float):
        student = student.astype(jnp.float32)
        teacher = teacher.astype(jnp.float32)
        student = student / jnp.sqrt(jnp.sum(jnp.square(student), axis=-1, keepdims=True) + 1e-6)
        teacher = teacher / jnp.sqrt(jnp.sum(jnp.square(teacher), axis=-1, keepdims=True) + 1e-6)

        logits = jnp.einsum("btd,ctd->btc", student, teacher) / jnp.asarray(temperature, dtype=jnp.float32)
        batch_size = logits.shape[0]
        labels = jnp.arange(batch_size)

        student_to_teacher = -jax.nn.log_softmax(logits, axis=-1)[labels, :, labels]
        teacher_to_student = -jax.nn.log_softmax(jnp.swapaxes(logits, 0, 2), axis=-1)[labels, :, labels]
        losses = 0.5 * (student_to_teacher + teacher_to_student)

        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    @staticmethod
    def _tokenwise_cosine_loss(student, teacher, token_mask):
        student = student.astype(jnp.float32)
        teacher = teacher.astype(jnp.float32)
        student = student / jnp.sqrt(jnp.sum(jnp.square(student), axis=-1, keepdims=True) + 1e-6)
        teacher = teacher / jnp.sqrt(jnp.sum(jnp.square(teacher), axis=-1, keepdims=True) + 1e-6)
        losses = 1.0 - jnp.sum(student * teacher, axis=-1)

        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    @staticmethod
    def _tokenwise_jepa_loss(student, teacher, token_mask):
        """Smooth L1 regression onto per-token LayerNorm teacher targets, as in I-JEPA."""
        student = student.astype(jnp.float32)
        teacher = teacher.astype(jnp.float32)
        teacher = jax.nn.standardize(teacher, axis=-1, epsilon=1e-5)

        absolute_error = jnp.abs(student - teacher)
        smooth_l1 = jnp.where(
            absolute_error < 1.0,
            0.5 * jnp.square(absolute_error),
            absolute_error - 0.5,
        )
        losses = jnp.mean(smooth_l1, axis=-1)
        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    def _tokenwise_distill_loss(self, student, teacher, token_mask):
        if self.distill_loss_type == "cosine":
            return self._tokenwise_cosine_loss(student, teacher, token_mask)
        if self.distill_loss_type == "jepa":
            return self._tokenwise_jepa_loss(student, teacher, token_mask)
        return self._tokenwise_contrastive_cosine_loss(
            student,
            teacher,
            token_mask,
            self.distill_contrast_temperature,
        )

    def _project_flow_distill(self, hidden, layer_ordinal: int):
        hidden = getattr(self, f"flow_distill_proj_in_{layer_ordinal}")(hidden)
        hidden = nnx.swish(hidden)
        return getattr(self, f"flow_distill_proj_out_{layer_ordinal}")(hidden)

    def _project_depth_distill(self, hidden, layer_ordinal: int):
        hidden = getattr(self, f"depth_distill_proj_in_{layer_ordinal}")(hidden)
        hidden = nnx.swish(hidden)
        return getattr(self, f"depth_distill_proj_out_{layer_ordinal}")(hidden)

    @staticmethod
    def _restore_aux_images(processed, original):
        updates = {}
        for key in ("flow_img", "wrist_flow_img", "depth_img", "wrist_depth_img"):
            value = getattr(original, key)
            if value is not None:
                updates[key] = value
        return processed if not updates else processed.replace(**updates)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False, train_progress=None):
        preprocess_rng, noise_rng, time_rng, query_noise_rng = jax.random.split(rng, 4)
        original_observation = observation
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, effort_type=self.effort_type)
        observation = self._restore_aux_images(observation, original_observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t_action = time_expanded * noise + (1 - time_expanded) * actions
        u_t_action = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        student_tokens, student_mask, student_ar_mask, student_adarms, noised_rate = self.embed_student_suffix(
            observation,
            x_t_action,
            time,
            train=train,
            noise_rng=query_noise_rng,
            train_progress=train_progress,
        )
        flow_tokens, flow_mask, flow_ar_mask, flow_adarms = self.embed_teacher_suffix(observation, x_t_action, time, "flow")
        depth_tokens, depth_mask, depth_ar_mask, depth_adarms = self.embed_teacher_suffix(
            observation, x_t_action, time, "depth"
        )

        outputs, layers = self._forward_all_streams(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            student_tokens,
            student_mask,
            student_ar_mask,
            student_adarms,
            flow_tokens,
            flow_mask,
            flow_ar_mask,
            flow_adarms,
            depth_tokens,
            depth_mask,
            depth_ar_mask,
            depth_adarms,
        )
        _, student_out, flow_out, depth_out = outputs

        student_v = self.action_out_proj_student(student_out[:, -self.action_horizon :])
        flow_v = self.action_out_proj_flow_teacher(flow_out[:, -self.action_horizon :])
        depth_v = self.action_out_proj_depth_teacher(depth_out[:, -self.action_horizon :])
        student_action_loss = jnp.mean(jnp.square(student_v - u_t_action), axis=(-2, -1))
        flow_teacher_action_loss = jnp.mean(jnp.square(flow_v - u_t_action), axis=(-2, -1))
        depth_teacher_action_loss = jnp.mean(jnp.square(depth_v - u_t_action), axis=(-2, -1))

        if self.action_only_pretrain:
            total_loss = (student_action_loss + flow_teacher_action_loss + depth_teacher_action_loss) / 3.0
            return total_loss, {
                "loss/student_action": student_action_loss,
                "loss/flow_teacher_action": flow_teacher_action_loss,
                "loss/depth_teacher_action": depth_teacher_action_loss,
                "noise/student_future_query_token_rate": jnp.mean(noised_rate),
                "noise/student_future_query_scale": self._student_query_noise_scale(train_progress),
                "loss/total": total_loss,
            }

        flow_slice = slice(1, 1 + self.flow_token_count)
        depth_slice = slice(1 + self.flow_token_count, 1 + self.flow_token_count + self.depth_token_count)
        def teacher_slice(count):
            return slice(1, 1 + count)
        flow_mask_tokens = jnp.ones((actions.shape[0], self.flow_token_count), dtype=jnp.bool_)
        depth_mask_tokens = jnp.ones((actions.shape[0], self.depth_token_count), dtype=jnp.bool_)
        flow_distill_losses = []
        depth_distill_losses = []
        for layer_ordinal, layer in enumerate(layers):
            _, student_hidden, flow_hidden, depth_hidden = layer
            student_flow_hidden = self._project_flow_distill(
                student_hidden[:, flow_slice, :], layer_ordinal
            )
            teacher_flow_hidden = jax.lax.stop_gradient(flow_hidden[:, teacher_slice(self.flow_token_count), :])
            flow_distill_losses.append(
                self._tokenwise_distill_loss(
                    student_flow_hidden,
                    teacher_flow_hidden,
                    flow_mask_tokens,
                )
            )
            student_depth_hidden = self._project_depth_distill(
                student_hidden[:, depth_slice, :], layer_ordinal
            )
            teacher_depth_hidden = jax.lax.stop_gradient(depth_hidden[:, teacher_slice(self.depth_token_count), :])
            depth_distill_losses.append(
                self._tokenwise_distill_loss(
                    student_depth_hidden,
                    teacher_depth_hidden,
                    depth_mask_tokens,
                )
            )

        future_flow_distill_loss = jnp.mean(jnp.stack(flow_distill_losses, axis=0), axis=0)
        future_depth_distill_loss = jnp.mean(jnp.stack(depth_distill_losses, axis=0), axis=0)
        total_loss = (
            self.student_action_loss_weight * student_action_loss
            + self.flow_teacher_action_loss_weight * flow_teacher_action_loss
            + self.depth_teacher_action_loss_weight * depth_teacher_action_loss
            + self.future_flow_contrast_loss_weight * future_flow_distill_loss
            + self.future_depth_contrast_loss_weight * future_depth_distill_loss
        )
        stats = {
            "loss/student_action": student_action_loss,
            "loss/flow_teacher_action": flow_teacher_action_loss,
            "loss/depth_teacher_action": depth_teacher_action_loss,
            f"loss/{self.distill_loss_type}_future_flow": future_flow_distill_loss,
            f"loss/{self.distill_loss_type}_future_depth": future_depth_distill_loss,
            "noise/student_future_query_token_rate": jnp.mean(noised_rate),
            "noise/student_future_query_scale": self._student_query_noise_scale(train_progress),
            "loss/total": total_loss,
        }
        return total_loss, stats

    def compute_loss(self, rng, observation, actions, *, train=False, train_progress=None):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train, train_progress=train_progress)
        return loss

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, debug_query_noise_scale=None):
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_noise_rng, query_noise_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None, None, None],
        )

        def step(carry):
            x_t, time, step_rng = carry
            step_rng, iter_query_noise_rng = jax.random.split(step_rng)
            student_tokens, student_mask, student_ar_mask, student_adarms, _ = self.embed_student_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                train=False,
                noise_rng=iter_query_noise_rng,
                train_progress=None,
                query_noise_scale=debug_query_noise_scale,
            )
            student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
            prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1
            outputs, _ = self.PaliGemma.llm(
                [None, student_tokens, None, None],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, student_adarms, None, None],
            )
            _, student_out, _, _ = outputs
            v_t = self.action_out_proj_student(student_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt, step_rng

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, query_noise_rng))
        return x_0
