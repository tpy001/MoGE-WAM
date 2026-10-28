"""Runtime helpers for RoboTwin evaluation; expert-cache replay is retired."""

import builtins


def fix_mplib_scene_reference():
    """Keep the existing planner memory fix while retaining full expert planners."""
    import envs.robot.robot as robot_module

    original = robot_module.MplibPlanner

    class SceneIndependentTimeParameterizer(original):
        def __init__(self, *args, **kwargs):
            import weakref

            super().__init__(*args, **kwargs)
            world = self.planner.planning_world
            world._sim_scene = weakref.proxy(world._sim_scene)

    robot_module.MplibPlanner = SceneIndependentTimeParameterizer


def quiet_step_progress():
    import envs._base_task as task_module

    def quiet_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith("step: \033[92m"):
            return
        builtins.print(*values, **kwargs)

    task_module.print = quiet_print
