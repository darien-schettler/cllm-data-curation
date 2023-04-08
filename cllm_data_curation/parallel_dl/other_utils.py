import pandas as pd
import pickle
import json
import os


def load_jsonl(path):
    """
    Load a jsonl file into a pandas dataframe
    """
    df = pd.read_json(path, lines=True,)
    return df


def save_pickle(obj: object, file_path: str) -> None:
    """
    Save an object to a pickle file.

    Args:
        obj: The object to save.
        file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> object:
    """
    Load an object from a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
