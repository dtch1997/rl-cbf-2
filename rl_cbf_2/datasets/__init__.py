import d4rl_slim as d4rl
import typing
import numpy as np
import logging

from rl_cbf_2.datasets.safety_conditions import get_safety_condition

D4RL_DATASETS = d4rl.list_datasets()

Dataset = typing.Dict[str, typing.Any]
DATASET_DESCRIPTIONS = {
    "mixed": ("expert", "medium", "random")
}

def is_custom_dataset(dataset_name: str) -> bool:
    """ Check if the dataset is a custom dataset """
    for key in DATASET_DESCRIPTIONS:
        if key in dataset_name:
            return True
    return False

def parse_compound_dataset_name(dataset_name: str) -> typing.Tuple[str, ...]:
    """ Parse a compound dataset name into the set of primitive datasets """
    env_type = dataset_name.split("-")[0]
    version = dataset_name.split("-")[-1]
    dataset_type = dataset_name[len(env_type) + 1 : -len(version) - 1]
    dataset_types = DATASET_DESCRIPTIONS[dataset_type]
    return tuple(f"{env_type}-{dataset_type}-{version}" for dataset_type in dataset_types)

def concatenate_datasets(datasets: typing.List[Dataset]) -> Dataset:    
    combined_dataset = {}
    for _dataset in datasets:
        for key, value in _dataset.items():
            if key not in combined_dataset:
                combined_dataset[key] = value
            else:
                if "metadata" in key:
                    # Fall back to the first dataset's metadata
                    continue
                combined_dataset[key] = np.concatenate((combined_dataset[key], value))
    return combined_dataset

def get_dataset(dataset_name) -> Dataset:

    if dataset_name in D4RL_DATASETS:
        return d4rl.get_dataset(dataset_name)
    
    # custom dataset
    elif is_custom_dataset(dataset_name):
        dataset_names = parse_compound_dataset_name(dataset_name)
        print(f"Loading datasets: {dataset_names}")
        datasets = [get_dataset(dataset_name) for dataset_name in dataset_names]
        return concatenate_datasets(datasets)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def relabel_dataset(dataset: Dataset, relabel_type: str, safety_fn) -> Dataset:
    """Relabel a dataset with a given safety function."""
    from rl_cbf_2.datasets.safety_reward import ZeroOneRewarder, IdentityRewarder
    if relabel_type == "identity":
        rewarder = IdentityRewarder(safety_fn)
    if relabel_type == "zero_one":
        rewarder = ZeroOneRewarder(safety_fn)
    else:
        raise ValueError(f"Unknown relabel type: {relabel_type}")
    dataset["rewards"] = rewarder.modify_reward(dataset["observations"], dataset["rewards"])
    return dataset    

def get_environment(dataset_name) -> "gymnasium.Environment":
    """Load a Gymnasium environment compatible with a given dataset."""

    if dataset_name in D4RL_DATASETS:
        return d4rl.get_environment(dataset_name)

    # custom dataset
    elif is_custom_dataset(dataset_name):
        dataset_names = parse_compound_dataset_name(dataset_name)
        return d4rl.get_environment(dataset_names[0])
    
def get_normalized_score(dataset_name, score):
    if dataset_name in D4RL_DATASETS:
        return d4rl.get_normalized_score(dataset_name, score)
    
    # custom dataset
    elif is_custom_dataset(dataset_name):
        dataset_names = parse_compound_dataset_name(dataset_name)
        return d4rl.get_normalized_score(dataset_names[0], score)