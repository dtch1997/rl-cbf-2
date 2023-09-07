""" Script to evaluate CQL with different safety thresholds."""

from absl import app

from rl_cbf_2.algos.cql_eval import *
from rl_cbf_2.algos.cql_eval import _CONFIG

import pandas as pd

def eval(_):
    config = _CONFIG.value
    env, actor, critic_1, critic_2 = load_model(config)

    # Evaluate CBF
    safety_thresholds = np.linspace(0.0, 1.0, 11)
    safe_episode_lengths = np.zeros_like(safety_thresholds)
    explore_fractions = np.zeros_like(safety_thresholds)

    for i, safety_threshold in enumerate(safety_thresholds):
        print("Safety threshold: ", safety_threshold)
        eval_metrics = eval_cbf(
            env, 
            actor, 
            critic_1,
            critic_2,
            device=config.device,
            n_episodes=config.n_episodes,
            seed=config.seed,
            safety_threshold=safety_threshold,
        )
        print(eval_metrics)
        safe_episode_lengths[i] = eval_metrics["episode_lengths"]
        explore_fractions[i] = eval_metrics["explore_fraction"]

    df = pd.DataFrame({
        "safety_threshold": safety_thresholds,
        "safe_episode_length": safe_episode_lengths,
        "explore_fraction": explore_fractions,
    })
    df.to_csv("cql_safety_threshold.csv")

if __name__ == "__main__":
    app.run(eval)