from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

# So we can see progress bars
tqdm.pandas()


def filter_parquet_file(pq_path, output_dir,
                        max_ll=600, min_len=50, min_max_ll=25, max_size_kbs=1_000,
                        min_alphanum=0.001, max_alphanum=0.975, min_ave_ll=16, min_lines=3, is_slim=True):
    """ Filter a Parquet file by line-length, file-size, number of lines, and alphanumerical fraction.

    Args:
        pq_path (str): The path to the Parquet file to filter.
        output_dir (str): The directory to save the filtered Parquet file.
        max_ll (int, optional): The maximum allowed line length.
        min_len (int, optional): The minimum allowed file size.
        min_max_ll (int, optional): The minimum allowed maximum line length.
        max_size_kbs (int, optional): The maximum allowed file size in kbs.
        min_alphanum (float, optional): The minimum allowed alphanumerical fraction.
        max_alphanum (float, optional): The maximum allowed alphanumerical fraction.
        min_ave_ll (int, optional): The minimum allowed average line length.
        min_lines (int, optional): The minimum allowed number of lines.
        is_slim (bool, optional): Whether to apply slim filtering or not.

    Returns:
        str: The path to the filtered Parquet file.
    """
    # Step 0 - Identify originating directory and create destination directory (if necessary)
    _root_path, _origin_root_dir, _lang, _fname = pq_path.rsplit("/", 3)
    _dest_dir = os.path.join(output_dir, _lang)
    _dest_path = pq_path.replace(_origin_root_dir, output_dir.rsplit("/", 1)[-1])
    if not os.path.isdir(_dest_dir):
        os.makedirs(_dest_dir, exist_ok=True)

    # Step 1 - Load the dataframe from the Parquet file (slim if necessary)
    _df = open_pq_as_df(pq_path, is_slim=is_slim)
    print(f"\t--> ORIGINAL LENGTH: {len(_df)}")

    # Step 2 - Filter out rows/files that have a maximum line length that falls outside
    #          the required range (min_max_ll <= max_ll <= max_line_len)
    _df = _df[((_df.max_ll <= max_ll) & (_df.max_ll >= min_max_ll))]
    print(f"\t--> AFTER MAX LINE-LENGTH REDUX: {len(_df)}")

    # Step 3 - Filter out rows/files that have a size (number of characters) less than `min_len`
    _df = _df[_df.file_size >= min_len]
    print(f"\t--> AFTER MIN FILE-SIZE REDUX: {len(_df)}")

    # Step 4 - Filter out rows/files that have a size greater than `max_size_kbs`
    _df = _df[_df.file_size//(1024**2) <= max_size_kbs]
    print(f"\t--> AFTER {max_size_kbs:,} KB MAX SIZE REDUX: {len(_df)}")

    # Step 5 - Filter out rows/files that have an alphanumeric fraction that
    #          falls outside the required range (min_alphanum, max_alphanum)
    _df = _df[((_df.alphanum_frac > min_alphanum) & (_df.alphanum_frac < max_alphanum))]
    print(f"\t--> AFTER ALPHANUMERIC REDUX: {len(_df)}")

    # Step 6 - Filter out rows/files that have an average line length less than `min_ave_ll`
    _df = _df[_df.ave_ll > min_ave_ll]
    print(f"\t--> AFTER MIN AVE LL REDUX: {len(_df)}")

    # Step 7 - Filter out rows/files that have fewer than `min_lines` lines
    _df = _df[(_df.file_size / _df.ave_ll) >= min_lines]
    print(f"\t--> AFTER MIN N-LINES REDUX: {len(_df)}")

    # Step 8 - Save the filtered dataframe to a Parquet file
    print(f"\t--> SAVING ...\n\t--> `{_dest_path}` ...\n")
    _df.to_parquet(_dest_path, index=False)

    # Step 9 - Return to sender
    return _dest_path


def open_pq_as_df(pq_path, is_slim=False):
    """Open a Parquet file as a Pandas DataFrame.

    Args:
        pq_path (str): The path to the Parquet file to open.
        is_slim (bool, optional):
            –  Whether the Parquet file is a slim version of the full dataset.
                --> If True, the full dataset columns will be read in, reduced and downcast
                --> If False, only the columns needed will be read in and are already downcasted
    Returns:
        _df (pd.DataFrame): The DataFrame containing the data from the Parquet file.
    """

    if is_slim:
        _df = pd.read_parquet(pq_path)

    else:
        # Load the pq file as a dataframe loading only columns we need
        _df = pd.read_parquet(pq_path, columns=["max_stars_repo_name", "ext", "content", "size", "max_line_length",
                                                "avg_line_length", "alphanum_fraction", "lang"])

        # Rename the columns
        _df.columns = ["repo_name", "file_ext", "content", "file_size", "max_ll",
                       "ave_ll", "alphanum_frac", "repo_lang"]

        # Downcast the columns to save memory (64 bit -> 32 bit)
        _df['file_size'] = _df['file_size'].astype(np.int32)
        _df['max_ll'] = _df['max_ll'].astype(np.int32)
        _df['ave_ll'] = _df['ave_ll'].astype(np.float32)
        _df['alphanum_frac'] = _df['alphanum_frac'].astype(np.float32)

    return _df


def flatten_l_o_l(nested_list):
    """Flatten a list of lists into a single list.

    Args:
        nested_list (list):
            – A list of lists (or iterables) to be flattened.

    Returns:
        list: A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional):
            – The symbol to use for the horizontal line
        line_len (int, optional):
            – The length of the horizontal line in characters
        newline_before (bool, optional):
            – Whether to print a newline character before the line
        newline_after (bool, optional):
            – Whether to print a newline character after the line
    """
    if newline_before: print()
    print(symbol * line_len)
    if newline_after: print()


def read_json_file(file_path):
    """Read a JSON file and parse it into a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict: A dictionary object representing the JSON data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the specified file path does not contain valid JSON data.
    """
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")


def glob_pq_paths(root_dir):
    """ Get all Parquet file paths in a directory. """
    def __check_capture(_path_list, _thresh=2):
        if len(_path_list) > _thresh:
            return True

    pattern_checks = [("**", "*.parquet"), ("*.parquet"), ("**", "**", "*.parquet")]
    for pattern in pattern_checks:
        pq_paths = glob(os.path.join(root_dir, *pattern))
        if __check_capture(pq_paths):
            return pq_paths
    else:
        raise FileNotFoundError(f"\nNo Parquet files found in {root_dir} based on pattern checks (see below)\n" 
                                f"PATTERN CHECKS:  {pattern_checks}")


def get_dir_size(path, unit='MB'):
    """
    Calculate the total size of a directory in the specified unit (KB, MB, or GB).

    Args:
        path (str): The path of the directory.
        unit (str, optional):
            – The unit of measurement for the size (default is 'MB' for megabytes).
                 --> Acceptable values are one of ['B', 'KB', 'MB', 'GB']

    Returns:
        float: The total size of the directory in the specified unit.
    """
    # Check if the provided path exists and is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a directory.")

    # Walk through the directory and sum the size of all files
    total_size = 0
    for dir_path, dir_names, filenames in os.walk(path):
        for f in filenames:
            file_path = os.path.join(dir_path, f)
            # Check if the file path is valid
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)

    # Convert the size based on the specified unit
    if unit == 'B':
        return total_size
    elif unit == 'KB':
        return total_size / 1024
    elif unit == 'MB':
        return total_size / (1024 ** 2)
    elif unit == 'GB':
        return total_size / (1024 ** 3)
    else:
        raise ValueError(f"Invalid unit '{unit}'. Accepted units are one of ['B', 'KB', 'MB', 'GB']")