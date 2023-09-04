"""Configuration for running rl_cbf_2.main """
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0

    # WandB parameters
    config.track = False
    config.wandb_project = "rl_cbf_2"
    config.wandb_entity = "dtch1997"
    config.wandb_name = None

    return config


def get_sweep(h):
    del h
    sweep = []
    for seed in [0, 1, 2]:
        sweep.append(
            {
                "config.seed": seed,
            }
        )
    return sweep