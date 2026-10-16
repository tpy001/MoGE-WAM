"""Five action-query groups with direct modality attention restrictions.

The shared, bidirectional VLM prefix can carry indirect modality information.
This is deliberately not strict missing-modality isolation.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos

ACTION_GROUPS = ("full", "no_head", "no_wrist", "no_language", "no_state")


def make_group_attention(prefix_mask, blocked_regions, horizon, group_indices):
    """Return suffix-to-prefix/suffix mask and repeated local RoPE positions."""
    visible = prefix_mask[:, None, :] & ~blocked_regions[:, jnp.asarray(group_indices), :]
    cross = jnp.repeat(visible, horizon, axis=1)
    ids = jnp.repeat(jnp.arange(len(group_indices)), horizon)
    within = ids[:, None] == ids[None, :]
    within = jnp.broadcast_to(within, (prefix_mask.shape[0], *within.shape))
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.tile(jnp.arange(horizon), len(group_indices))[None]
    return jnp.concatenate([cross, within], axis=-1), positions


class Pi0MultiAction(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = gemma.get_config(config.action_expert_variant).width
        embeddings = 0.02 * jax.random.normal(rngs.params(), (len(ACTION_GROUPS), width))
        self.action_group_embeddings = nnx.Param(embeddings.at[0].set(0))

    def _embed_multi_prefix(self, observation):
        if observation.tokenized_prompt_language_mask is None or observation.tokenized_prompt_state_mask is None:
            raise ValueError("Multi-action requires language/state token region masks.")
        # Use Pi0's standard embeddings, keeping checkpoint parameter paths intact.
        tokens, valid, ar_mask = self.embed_prefix(observation)
        text_length = observation.tokenized_prompt.shape[1]
        regions = jnp.zeros((valid.shape[0], len(ACTION_GROUPS), valid.shape[1]), dtype=jnp.bool_)
        offset = 0
        for name in observation.images:
            # SigLIP has the same patch-token count for every preprocessed image.
            image_length = (tokens.shape[1] - text_length) // len(observation.images)
            group = {"base_0_rgb": 1, "left_wrist_0_rgb": 2}.get(name)
            if group is not None:
                regions = regions.at[:, group, offset : offset + image_length].set(True)
            offset += image_length
        regions = regions.at[:, 3, offset:].set(observation.tokenized_prompt_language_mask)
        regions = regions.at[:, 4, offset:].set(observation.tokenized_prompt_state_mask)
        return tokens, valid, ar_mask, regions

    def _embed_groups(self, observation, noisy_actions, time, group_indices):
        batch, groups, horizon, dim = noisy_actions.shape
        flat_actions = noisy_actions.reshape(batch, groups * horizon, dim)
        del observation  # pi05 state is already encoded in the prefix.
        tokens = self.action_in_proj(flat_actions)
        time_emb = posemb_sincos(time, tokens.shape[-1], min_period=4e-3, max_period=4.0)
        cond = nnx.swish(self.time_mlp_out(nnx.swish(self.time_mlp_in(time_emb))))
        embeddings = self.action_group_embeddings.value[jnp.asarray(group_indices)].astype(tokens.dtype)
        tokens = tokens.reshape(batch, groups, horizon, -1) + embeddings[None, :, None, :]
        return tokens.reshape(batch, groups * horizon, -1), cond

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions
        group_indices = tuple(range(len(ACTION_GROUPS)))
        grouped = jnp.broadcast_to(x_t[:, None], (x_t.shape[0], len(ACTION_GROUPS), *x_t.shape[1:]))
        prefix, valid, ar_mask, regions = self._embed_multi_prefix(observation)
        suffix, cond = self._embed_groups(observation, grouped, time, group_indices)
        suffix_mask, suffix_positions = make_group_attention(valid, regions, self.action_horizon, group_indices)
        prefix_mask = make_attn_mask(valid, ar_mask)
        prefix_mask = jnp.pad(prefix_mask, ((0, 0), (0, 0), (0, suffix.shape[1])))
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        positions = jnp.concatenate([jnp.cumsum(valid, axis=1) - 1, suffix_positions], axis=1)
        (_, output), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=mask,
            positions=positions,
            adarms_cond=[None, cond],
        )
        velocity = self.action_out_proj(output).reshape(grouped.shape)
        per_group = jnp.mean(jnp.square(velocity - target[:, None]), axis=-1)
        loss = jnp.sum(per_group, axis=1)
        stats = {f"loss/action_{name}": jnp.mean(per_group[:, i], axis=-1) for i, name in enumerate(ACTION_GROUPS)}
        stats["loss/total"] = jnp.mean(loss, axis=-1)
        return loss, stats

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_loss_with_stats(rng, observation, actions, train=train)[0]

    def _sample_groups(self, rng, observation, group_indices, *, num_steps, noise):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        observation = model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
        if noise.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError("noise must have shape [B, H, D].")
        grouped = jnp.broadcast_to(noise[:, None], (batch, len(group_indices), *noise.shape[1:]))
        prefix, valid, ar_mask, regions = self._embed_multi_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(valid, ar_mask),
            positions=jnp.cumsum(valid, axis=1) - 1,
        )
        mask, positions = make_group_attention(valid, regions, self.action_horizon, group_indices)
        dt = -1.0 / num_steps

        def step(index, x_t):
            time = jnp.full((batch,), 1.0 + index * dt)
            suffix, cond = self._embed_groups(observation, x_t, time, group_indices)
            (_, output), _ = self.PaliGemma.llm(
                [None, suffix],
                mask=mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, cond],
            )
            velocity = self.action_out_proj(output).reshape(x_t.shape)
            return x_t + dt * velocity

        return jax.lax.fori_loop(0, num_steps, step, grouped)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, group="full"):
        if group not in ACTION_GROUPS:
            raise ValueError(f"Unknown action group {group!r}; expected one of {ACTION_GROUPS}.")
        return self._sample_groups(rng, observation, (ACTION_GROUPS.index(group),), num_steps=num_steps, noise=noise)[
            :, 0
        ]

    def sample_actions_all(self, rng, observation, *, num_steps=10, noise=None):
        actions = self._sample_groups(
            rng, observation, tuple(range(len(ACTION_GROUPS))), num_steps=num_steps, noise=noise
        )
        return {name: actions[:, i] for i, name in enumerate(ACTION_GROUPS)}
