"""Defer SAPIEN render synchronization until a policy observation is requested."""

import time
import weakref


class ObservationRenderRuntime:
    def __init__(self, env, *, enabled=True):
        self.env = env
        self.enabled = enabled
        self.observing = False
        self.in_action = False
        self.in_expert = False
        self.stats = {
            "render_syncs": 0,
            "suppressed_render_syncs": 0,
            "observations": 0,
            "suppressed_terminal_observations": 0,
            "observation_seconds": 0.0,
            "action_seconds": 0.0,
            "expert_seconds": 0.0,
            "setup_seconds": 0.0,
            "deferred_camera_updates": 0,
        }
        self.original_setup_scene = env.setup_scene
        self.original_setup_demo = env.setup_demo
        self.original_get_obs = env.get_obs
        self.original_take_action = env.take_action
        self.original_play_once = env.play_once
        env.setup_scene = self.setup_scene
        env.setup_demo = self.setup_demo
        env.get_obs = self.get_obs
        env.take_action = self.take_action
        env.play_once = self.play_once
        self.original_update_render = getattr(env, "_update_render", None)
        if self.original_update_render is not None:
            env._update_render = self.update_render  # noqa: SLF001 -- Per-instance simulator hook.

    def update_render(self):
        if not self.enabled or self.observing or self.env.crazy_random_light:
            # Crazy lighting must keep its original RNG and light-state updates.
            return self.original_update_render()
        self.stats["deferred_camera_updates"] += 1
        return None

    def setup_scene(self, *args, **kwargs):
        result = self.original_setup_scene(*args, **kwargs)
        # Do not keep the scene alive through its own instance callback.
        update_render = weakref.WeakMethod(self.env.scene.update_render)

        def gated_update():
            if not self.enabled or self.observing:
                self.stats["render_syncs"] += 1
                original_update_render = update_render()
                if original_update_render is not None:
                    return original_update_render()
                return None
            self.stats["suppressed_render_syncs"] += 1
            return None

        self.env.scene.update_render = gated_update
        return result

    def setup_demo(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.original_setup_demo(*args, **kwargs)
        finally:
            self.stats["setup_seconds"] += time.perf_counter() - started

    def get_obs(self):
        if self.enabled and (self.in_action or self.in_expert):
            self.stats["suppressed_terminal_observations"] += 1
            return self.env.now_obs
        started = time.perf_counter()
        self.observing = True
        try:
            self.stats["observations"] += 1
            return self.original_get_obs()
        finally:
            self.observing = False
            self.stats["observation_seconds"] += time.perf_counter() - started

    def take_action(self, *args, **kwargs):
        started = time.perf_counter()
        self.in_action = True
        try:
            return self.original_take_action(*args, **kwargs)
        finally:
            self.in_action = False
            self.stats["action_seconds"] += time.perf_counter() - started

    def play_once(self, *args, **kwargs):
        started = time.perf_counter()
        self.in_expert = True
        try:
            return self.original_play_once(*args, **kwargs)
        finally:
            self.in_expert = False
            self.stats["expert_seconds"] += time.perf_counter() - started
