import pandas as pd
from multiprocessing import cpu_count


def get_repos(local_path=None,
              frac=1.,
              seg_num=1,
              remote_path='gs://kds-c1baa21e1604b0095451b700a1595f4a75ade1d1289d6a66cf9b3f31/no_over__all_data (1).csv',
              as_list=True):
    """Get a list of repos to process.

    Args:
        local_path (str, optional): Path to local csv file
        frac (float, optional): Fraction of repos to use
        seg_num (int, optional): Segment number to use
        remote_path (str, optional): Path to remote csv file
        as_list (bool, optional): Return as list or dataframe

    Returns:
        list: List of repos to process
    """
    if local_path:
        df = pd.read_csv(local_path, usecols=['repo_name'])['repo_name']
    else:
        df = pd.read_csv(remote_path, usecols=['repo_name'])['repo_name']
        if frac < 1.0:
            if seg_num is None:
                df = df.sample(frac=frac).reset_index(drop=True)
            else:
                frac_n = int(len(df) * frac)
                df = df.iloc[frac_n*(seg_num-1):frac_n*seg_num].reset_index(drop=True)
    return df.to_list() if as_list else df


def get_bad_extensions(additional_exts=None):
    """Get a list of bad extensions to filter out.

    Args:
        additional_exts (list, optional): Additional extensions to filter out

    Returns:
        list: List of bad extensions
    """

    bad_ext = [
        '3gp', 'aac', 'aif', 'aiff', 'amr', 'app', 'au', 'avi', 'bin', 'bmp', 'bz2', 'class', 'csv', 'dat', 'db', 'dll',
        'dng', 'dylib', 'egg', 'eot', 'exe', 'flac', 'flv', 'gif', 'gitignore', 'glif', 'gradle', 'gz', 'heic', 'heif',
        'ico', 'jar', 'jpeg', 'jpg', 'lo', 'lock', 'log', 'm4a', 'm4v', 'mid', 'midi', 'mkv', 'mov', 'mp3', 'mp4',
        'mpeg', 'mpg', 'nar', 'o', 'ogg', 'ogv', 'opus', 'otf', 'p', 'pdf', 'pickle', 'pkl', 'png', 'pyc', 'pyd', 'pyo',
        'ra', 'ram', 'rkt', 'rm', 'so', 'ss', 'svg', 't3', 'tar', 'tif', 'tiff', 'ts', 'tsv', 'ttf', 'war', 'wav',
        'webm', 'webp', 'wmv', 'woff', 'woff2', 'xz', 'zip', 'zst',
    ]
    if additional_exts:
        bad_ext.extend(additional_exts)
    return bad_ext


def get_parallel_params(overrides=None):
    """Get parameters for parallel processing """
    param_map = dict(
        n_threads=cpu_count() * 3,
        chunk_size=cpu_count() * 3 * 3,
    )
    return param_map if not overrides else {**param_map, **overrides}


def get_repo_chunks(repo_list, chunk_size):
    """ Split a list of repos into chunks of size chunk_size """
    return [repo_list[i:i + chunk_size] for i in range(0, len(repo_list), chunk_size)]


def get_mime_type(path, _mime):
    """Get the mime type of file """
    return _mime.from_file(path)


def print_check(n_threads, chunk_size, repo_list, repo_chunks, n_rejects):
    """Print out some constants to check that they are what we expect"""
    print("... CONSTANTS:")
    print(f"\tn_threads           --> {n_threads}"
          f"\n\tchunk_size          --> {chunk_size}"
          f"\n\tlen(repo_list)      --> {len(repo_list)}"
          f"\n\tlen(repo_chunks)    --> {len(repo_chunks)}"
          f"\n\tlen(bad_extensions) --> {n_rejects}"
          f"\n\tlen(repo_chunks[0]) --> {len(repo_chunks[0])}\n")
