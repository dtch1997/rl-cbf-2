import d4rl_slim as d4rl
import typing
import numpy as np
import logging

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
            print(key, value.shape)
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