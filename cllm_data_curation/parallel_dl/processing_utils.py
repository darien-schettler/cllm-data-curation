import os
import copy
import magic
import shutil
import chardet
import traceback
import subprocess

from cllm_data_curation.parallel_dl.preprocessing_utils import get_bad_extensions


class TimeoutError(Exception):
    """Custom exception class to be raised when a timeout occurs."""
    pass


def timeout(func, args=(), kwargs=None, timeout_duration=150, default=None):
    """
    Wraps a function and enforces a timeout duration. If the function doesn't complete
    within the specified duration, a TimeoutError is raised, and the default value is
    returned.

    Args:
        func (callable): The function to wrap.
        args (tuple): Positional arguments to be passed to the function.
        kwargs (dict, optional): Keyword arguments to be passed to the function.
        timeout_duration (int, optional): Timeout duration in seconds. Defaults to 150.
        default (Any, optional): Default value to be returned if a timeout occurs.

    Returns:
        Any: The result of the function call, or the default value in case of a timeout.
    """
    import signal

    def handler(signum, frame):
        """Signal handler to raise a TimeoutError."""
        raise TimeoutError()

    # Set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)

    if kwargs is None:
        kwargs = {}

    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = default
    finally:
        signal.alarm(0)

    return result


def get_content(file_path, mime_obj):
    """ Reads the content of a file and returns it as a string

    Args:
        file_path (str): Path to the file.
        mime_obj (magic.Magic): Magic object for determining file types.

    Returns:
        str: Content of the file, or None if the file could not be read.
    """

    def read_file_with_encoding(_file_path, encoding):
        """Reads a file using the specified encoding.

        Args:
            _file_path (str): Path to the file.
            encoding (str): Encoding to use when reading the file.

        Returns:
            str: Content of the file, or None if the file could not be read.
        """
        try:
            with open(file_path, 'rb') as f:
                _content = f.read()
            return _content.decode(encoding)
        except UnicodeDecodeError:
            return None

    # Check if the file is a text file
    if not mime_obj.from_file(file_path).startswith('text'):
        return None

    # Attempt to read the file using UTF-8 encoding
    content = read_file_with_encoding(file_path, 'UTF-8')

    # If UTF-8 failed, try to detect the encoding and read the file again
    if content is None:
        with open(file_path, 'rb') as file_handle:
            content_bytes = file_handle.read()
        encoding_info = chardet.detect(content_bytes)

        if encoding_info['encoding'] is not None:
            content = read_file_with_encoding(file_path, encoding_info['encoding'])

    # Check if the content should be kept
    if content is not None and keep(content):
        return content
    else:
        return None


def process_repo(repo_data, repo_dir, bad_exts, _mime):
    """Processes a single repo and returns a list of files and their metadata.

    Args:
        repo_data (str): Name of the repo.
        repo_dir (str): Path to the repo.
        bad_exts (list): List of extensions to ignore.
        _mime (magic.Magic): Magic object for determining file types.

    Returns:
        list: List of tuples of the form (file, metadata).
    """

    def _is_valid_file(_file_path):
        return (
                '.git' not in _file_path and
                _file_path[0] != '.' and
                'LICENSE' not in _file_path and
                'node_modules' not in _file_path and
                '.min.' not in _file_path and
                _file_path.split('.')[-1] not in bad_exts
        )

    def _get_extensions(_files):
        exts = []
        for _file_path in _files:
            try:
                exts.append(_mime.from_file(_file_path))
            except FileNotFoundError:
                exts.append("n/a")
        return exts

    _mime = magic.Magic(mime=True)

    output = []
    meta = {'repo_name': repo_data}

    try:
        for current_dir, _, files in os.walk(repo_dir):
            valid_files = [os.path.join(current_dir, f) for f in files if _is_valid_file(f)]
            filenames = [f.replace(repo_dir + '/', '') for f in valid_files]
            extensions = _get_extensions(valid_files)

            text_outputs = []
            for file_path in valid_files:
                try:
                    text_outputs.append(get_content(file_path, _mime))
                except TimeoutError:
                    raise
                except Exception:
                    text_outputs.append(None)

            for i, text in enumerate(text_outputs):
                if text is not None:
                    meta_ind = copy.deepcopy(meta)
                    meta_ind['file_name'] = filenames[i]
                    meta_ind['mime_type'] = extensions[i]
                    output.append([text, meta_ind])

        shutil.rmtree(repo_dir, ignore_errors=True)

    except TimeoutError:
        print(f"Processing for {repo_data} timed out")

    return output


def get_file_size(len_text, b_format="MB"):
    """calculate size of text from string assuming 1 character is 1 byte """
    if b_format == "KB":
        return len_text / 1024
    elif b_format == "MB":
        return len_text / (1024 ** 2)  # default
    elif b_format == "GB":
        return len_text / (1024 ** 3)
    else:
        return len_text  # bytes


def keep(x, max_mb=1, min_n_chars=16):
    """ overwrite previous implementation from author and only do minimal filtering

    Args:
        x (str): text to be checked
        max_mb (int, optional): maximum size of text in MB. Defaults to 1.
        min_n_chars (int, optional): minimum number of characters. Defaults to 16.

    Returns:
        bool: True if text is kept, False otherwise
    """
    return False if (get_file_size(len(x)) > max_mb) or (len(x) < min_n_chars) else True


def process_repo_list(repo_data, clone_timeout, _mime, _tmp_dir=".tmp"):
    """ Processes a list of repos and returns a list of files and their metadata

    Args:
        repo_data (str): repo to process
        clone_timeout (int): timeout for cloning a repo
        _mime (magic.Magic): Magic object for determining file types.
        _tmp_dir (str, optional): path to temporary directory. Defaults to ".tmp".

    Returns:
        list: list of tuples of the form (file, metadata)
    """

    # Get bad extensions to filter out
    _bad_exts = get_bad_extensions()

    if not os.path.isdir(_tmp_dir):
        os.makedirs(_tmp_dir, exist_ok=True)
    try:
        # Get repo directory path (hidden directory with repo name)
        repo_dir = os.path.join(_tmp_dir, repo_data.rsplit("/", 1)[-1])

        # clones master branch of repos with depth 1 (most recent commit only), ignoring any terminal prompts
        p = subprocess.Popen(
            f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 --single-branch https://github.com/{repo_data} {repo_dir}',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        try:
            p.wait(clone_timeout)
        except subprocess.TimeoutExpired:
            print(f'Git clone for {repo_data} timed out ')
            p.kill()
        shutil.rmtree(os.path.join(repo_dir, '.git'), ignore_errors=True)

        # extracts text files from repo and returns them as list : [[text, metadata], ... ]
        out = process_repo(repo_data, repo_dir, _bad_exts, _mime)  # , processing_timeout=processing_timeout)
    except Exception:
        print(traceback.format_exc())
        out = None
    return out
