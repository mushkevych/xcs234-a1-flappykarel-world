from typing import Dict

from flappy_karel_env import FlappyKarelEnv, REWARD_SCHEDULE_A, REWARD_SCHEDULE_B, REWARD_SCHEDULE_C, REWARD_SCHEDULE_D
from flappy_karel_env import MAP_NAME_7x5_S3, MAP_NAME_7x5_S2
from submission import policy_iteration, value_iteration, render_single


def run_policy_iteration(map_name: str, reward_schedule: Dict[bytes, float], are_borders_present: bool):
    """Visualize policy iteration on FlappyKarel deterministic environment"""
    env = FlappyKarelEnv(
        map_name=map_name, reward_schedule=reward_schedule, are_borders_present=are_borders_present, render_mode="human"
    )

    V_pi, p_pi = policy_iteration(
        env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-3
    )
    render_single(env, p_pi, 100)


def run_value_iteration(map_name: str, reward_schedule: Dict[bytes, float], are_borders_present: bool):
    """Visualize policy iteration on FlappyKarel deterministic environment"""
    env = FlappyKarelEnv(
        map_name=map_name, reward_schedule=reward_schedule, are_borders_present=are_borders_present, render_mode="human"
    )

    V_pi, p_pi = value_iteration(
        env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-3
    )
    render_single(env, p_pi, 100)


if __name__ == '__main__':
    run_policy_iteration(map_name=MAP_NAME_7x5_S2, reward_schedule=REWARD_SCHEDULE_D, are_borders_present=True)
    # run_value_iteration(reward_schedule=REWARD_SCHEDULE_B, are_borders_present=True)
