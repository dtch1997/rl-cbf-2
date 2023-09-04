import numpy as np

def get_safety_condition(env_name: str):
    if "walker2d" in env_name:
        return walker2d_is_unsafe
    elif "ant" in env_name:
        return ant_is_unsafe
    elif "hopper" in env_name:
        return hopper_is_unsafe
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def walker2d_is_unsafe(states: np.ndarray):
    height = states[..., 0]
    angle = states[..., 1]

    height_ok = np.logical_and(height > 0.8, height < 2.0)
    angle_ok = np.logical_and(angle > -1.0, angle < 1.0)
    is_safe = np.logical_and(height_ok, angle_ok)
    return (~is_safe).astype(float)

def ant_is_unsafe(states: np.ndarray):
    height = states[..., 0]
    is_safe = np.logical_and(height > 0.2, height < 1.0)
    return (~is_safe).astype(float)

def hopper_is_unsafe(states: np.ndarray):
    height = states[..., 0]
    ang = states[..., 1]
    remainder = states[..., 2:]
    is_safe = np.logical_and(height > 0.7, np.abs(ang) < 0.2)
    is_safe = np.logical_and(is_safe, (np.abs(remainder) < 100.0).all())
    return (~is_safe).astype(float)