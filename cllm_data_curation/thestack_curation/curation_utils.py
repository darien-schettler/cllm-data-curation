from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import ast
import os
import re

# So we can see progress bars
tqdm.pandas()


def make_meta_df(root_dir):
    """ Make a DataFrame containing metadata about the Parquet files in the specified directory.

    Args:
        root_dir (str): The path to the root directory containing the Parquet files.

    Returns:
        pd.DataFrame: A DataFrame containing metadata about the Parquet files in the specified directory.
    """
    all_pq_paths = glob_pq_paths(root_dir)
    meta_df = pd.DataFrame({"pq_path": all_pq_paths})
    meta_df["lang"] = meta_df.pq_path.apply(lambda x: x.rsplit("/", 2)[-2])
    lsizes = meta_df.groupby("lang")["pq_path"].first().apply(lambda x: get_dir_size(os.path.dirname(x))).to_dict()
    meta_df["lang_size_mb"] = meta_df["lang"].map(lsizes)
    lcnts = meta_df.groupby("lang")["pq_path"].first().apply(lambda x: len(os.listdir(os.path.dirname(x)))).to_dict()
    meta_df["lang_file_cnt"] = meta_df["lang"].map(lcnts)
    return meta_df


def filter_meta_languages(meta_df, top_k=None, mb_size_thresh=None, pq_file_cnt_thresh=None, bad_langs=(".csv",)):
    """ Filter the DataFrame containing metadata about the Parquet files in the specified directory.

    Args:
        meta_df (pd.DataFrame):
            – The DataFrame containing metadata about the Parquet files in the specified directory.
        top_k (int, optional):
            – The number of languages to keep.
              If specified, this will override `mb_size_thresh` and `pq_file_cnt_thresh`.
        mb_size_thresh (float, optional):
            – The minimum size of the language in megabytes.
            – If specified, this will override `pq_file_cnt_thresh`.
        pq_file_cnt_thresh (int, optional):
            – The minimum number of Parquet files in the language.
        bad_langs (tuple, optional):
            – A tuple of languages to exclude from the DataFrame no matter what.
            – This is useful for filtering out languages that are not actually languages (e.g. ".csv").

    Returns:
        pd.DataFrame: A DataFrame containing metadata about the Parquet files in the specified directory.
    """
    if top_k:
        top_langs = meta_df.groupby("lang")["lang_size_mb"].sum().sort_values(ascending=False).index[:top_k]
        meta_df = meta_df[meta_df.lang.isin(top_langs)]
    elif mb_size_thresh:
        meta_df = meta_df[meta_df.lang_size_mb > mb_size_thresh]
    elif pq_file_cnt_thresh:
        meta_df = meta_df[meta_df.lang_file_cnt >= pq_file_cnt_thresh]
    return meta_df[~meta_df.lang.isin(bad_langs)].reset_index(drop=True)


def filter_parquet_file(pq_path, output_dir,
                        max_ll=600, min_len=50, min_max_ll=25, max_size_kbs=1_000, keep_original_index=True,
                        min_alphanum=0.001, max_alphanum=0.975, min_ave_ll=16, min_lines=3, is_slim=True, **kwargs):
    """Filter a Parquet file by applying multiple criteria such as line length, file size,
    number of lines, and alphanumerical fraction.

    Args:
        pq_path (str): Path to the input Parquet file.
            - NOTE: This is expected to be in the form: .../<root_dir>/<lang>/<pq_file_name>.pq
        output_dir (str): Directory to save the filtered Parquet file.
        max_ll (int, optional): Maximum allowed line length. Defaults to 600.
        min_len (int, optional): Minimum allowed file size. Defaults to 50.
        min_max_ll (int, optional): Minimum allowed maximum line length. Defaults to 25.
        max_size_kbs (int, optional): Maximum allowed file size in kilobytes. Defaults to 1000.
        keep_original_index (bool, optional): Whether to keep the original index. Defaults to True.
        min_alphanum (float, optional): Minimum allowed alphanumerical fraction. Defaults to 0.001.
        max_alphanum (float, optional): Maximum allowed alphanumerical fraction. Defaults to 0.975.
        min_ave_ll (int, optional): Minimum allowed average line length. Defaults to 16.
        min_lines (int, optional): Minimum allowed number of lines. Defaults to 3.
        is_slim (bool, optional): Whether to apply slim filtering or not. Defaults to True.

    Returns:
        str: Path to the filtered Parquet file.
    """
    print(f"\nWORKING ON {pq_path} ...")

    # Step 0: Identify input directory and create destination directory (if necessary)
    _root_path, _origin_root_dir, _lang, _fname = pq_path.rsplit("/", 3)
    _dest_dir = os.path.join(output_dir, _lang)
    _dest_path = pq_path.replace(_origin_root_dir, output_dir.rsplit("/", 1)[-1])
    if not os.path.isdir(_dest_dir):
        os.makedirs(_dest_dir, exist_ok=True)

    # Step 1: Load the DataFrame from the Parquet file (slim if necessary)
    _df = open_pq_as_df(pq_path, is_slim=is_slim)
    print(f"\t--> ORIGINAL LENGTH: {len(_df)}")

    # Step 2: Filter out rows/files with maximum line lengths outside the required range
    filter_flag = ((_df.max_ll <= max_ll) & (_df.max_ll >= min_max_ll))
    reject_df = _df[~filter_flag].copy()
    reject_df.loc[:, "reason"] = "max_ll"
    _df = _df[filter_flag]
    print(f"\t--> AFTER MAX LINE-LENGTH REDUCTION: {len(_df)}")

    # Step 3: Filter out rows/files with sizes smaller than `min_len`
    filter_flag = _df.file_size >= min_len
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("file_too_small")
    _df = _df[filter_flag]
    print(f"\t--> AFTER MIN FILE-SIZE REDUCTION: {len(_df)}")

    # Step 4: Filter out rows/files with sizes larger than `max_size_kbs`
    filter_flag = _df.file_size // 1024 <= max_size_kbs
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("file_too_large")
    _df = _df[filter_flag]
    print(f"\t--> AFTER {max_size_kbs:,} KB MAX SIZE REDUCTION: {len(_df)}")

    # Step 5: Filter out rows/files with an alphanumeric fraction outside the required range
    filter_flag = ((_df.alphanum_frac > min_alphanum) & (_df.alphanum_frac < max_alphanum))
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("alphanum_frac")
    _df = _df[filter_flag]
    print(f"\t--> AFTER ALPHANUMERIC REDUCTION: {len(_df)}")

    # Step 6: Filter out rows/files with an average line length smaller than `min_ave_ll`
    filter_flag = _df.ave_ll > min_ave_ll
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("ave_ll")
    _df = _df[filter_flag]
    print(f"\t--> AFTER MIN AVERAGE LINE LENGTH REDUCTION: {len(_df)}")

    # Step 7: Filter out rows/files with fewer than `min_lines` lines
    filter_flag = (_df.file_size / _df.ave_ll) >= min_lines
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("min_lines")
    _df = _df[filter_flag]
    print(f"\t--> AFTER MIN NUMBER OF LINES REDUCTION: {len(_df)}")

    # Step 8: Filter out rows/files written in Python 2
    filter_flag = _df.content.apply(test_source_code_compatible)
    reject_df = pd.concat((reject_df, _df[~filter_flag].copy()))
    reject_df.loc[:, "reason"] = reject_df["reason"].fillna("python2")
    _df = _df[filter_flag]
    print(f"\t--> AFTER PYTHON 2 DETECTION REDUCTION: {len(_df)}")

    # Step 9: Save the filtered DataFrame to a Parquet file
    print(f"\t--> SAVING ...\n\t--> `{_dest_path}` ...\n")
    _df.to_parquet(_dest_path, index=keep_original_index)

    # Step 9.5: Save the rejection DataFrame to a Parquet file
    reject_df = reject_df.sort_index()  # sort to reorder according to the original ordering
    reject_df.to_parquet(_dest_path.replace(".parquet", "_rejects.parquet"), index=keep_original_index)

    # Step 10: Return the path to the filtered Parquet file
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

    pattern_checks = [("**", "*.parquet"), ("*.parquet",), ("**", "**", "*.parquet")]
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
    if unit.upper() == 'B':
        return total_size
    elif unit.upper() == 'KB':
        return total_size / 1024
    elif unit.upper() == 'MB':
        return total_size / (1024 ** 2)
    elif unit.upper() == 'GB':
        return total_size / (1024 ** 3)
    else:
        raise ValueError(f"Invalid unit '{unit}'. Accepted units are one of ['B', 'KB', 'MB', 'GB']")


def replace_byte_encoded_string(input_string, min_length=100, replacement_token="<BYTE_ENCODED_STRING>"):
    """ Replace a byte-encoded string with a token. """
    # Pre-compile the pattern for better performance
    #    --> the following pattern matches a byte-encoded string of at least `min_length`
    pattern = re.compile(fr"b'([^\x00-\x7F]{{{min_length},}})'")

    # Replace the byte-encoded string with the replacement token
    replaced_string = pattern.sub(replacement_token, input_string)

    return replaced_string


def contains_repeating_substring(input_string, substring, n):
    """ Check if a string contains a substring that repeats at least n times. """
    # Escape the substring to prevent regex errors and build the pattern
    pattern = f"({re.escape(substring)}){{{n},}}"

    # Check if the pattern is found in the input string and return flag
    return bool(re.search(pattern, input_string))


def test_source_code_compatible(input_string):
    """
    Test if the input source code is compatible with the current Python version.
    This function has some limitations:
    - It will only test compatibility with the Python version you are running.
    - It won't differentiate between syntax errors and incompatibilities.

    Args:
        code_data (str): The source code to test compatibility.

    Returns:
        bool: If incompatible, returns False else True
    """
    try:
        # Try to parse the source code as an Abstract Syntax Tree (AST)
        # If it succeeds, the code is compatible with the current Python version
        _ = ast.parse(input_string)
        return True
    except SyntaxError as exc:
        # If a SyntaxError occurs, it means the code is not compatible with the current Python version
        return False
    except ValueError as exc:
        # If a ValueError occurs, it means there is an issue with the input and the code is not compatible
        return False


####################################################################################################
# TODO - Create a function that allows us to visualize the filtered files and
#        explains why the files were filtered out in the first place
####################################################################################################
# def print_demo_omits(lang, n_demo_samples=10, max_print_ln=5000):
#     print(f"\n{'='*100}\n\t\t\t{n_demo_samples} DEMO EXMAPLES FOR LANGUAGE --> {lang} \n{'='*100}\n")
#     paths = lang_to_example_map[lang]
#     orig_path = paths["full_pq_path"]
#     slim_v1_path = paths["clean_pq_path"]
#     _orig_df = open_pq_as_df(orig_path)
#     _slim_v1_df = open_pq_as_df(slim_v1_path, is_slim=True)
####################################################################################################
