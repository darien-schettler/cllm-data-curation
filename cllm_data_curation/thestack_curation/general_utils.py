import os
import csv
import subprocess


def authenticate_hf(hf_token):
    """ Authenticate with HuggingFace using the specified token. """

    # If this fails you may need to install huggingface_hub
    # and/or add the huggingface-cli to your PATH
    #
    # In my case I had to add the following line to my ~/.bashrc
    # `export PATH="/Users/<user>/Library/Python/<python-version>/bin/:$PATH"`
    #
    # I guess my bin directory wasn't in my PATH for some reason
    subprocess.call(["huggingface-cli", "login", "--token", hf_token])


def get_optimal_worker_count(worker_factor=1, fallback_n_workers=4):
    """ Return the optimal number of workers to use for parallel processing.

    Args:
        worker_factor (int, optional):
            – the factor to multiply the number of CPUs by
        fallback_n_workers (int, optional):
            – the number of workers to use if the number of CPUs cannot be determined
    Returns:
        cpu_count (int): the 'optimal' number of workers to use for parallel processing
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        # Fallback to a reasonable default if the function returns None
        return fallback_n_workers
    else:
        # Adjust this factor as needed based on network conditions and other factors
        return cpu_count * worker_factor


def read_csv_urls(file_path):
    """Read a single column CSV file into a list.

    Args:
        file_path (str): the path to the CSV file.

    Returns:
        data (list): The list of URLs for all parquet files in the dataset
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [row[0] for row in reader]
    return data[1:]  # skip the header row


def get_abs_current_file_path():
    """Return the absolute path of the project root directory."""
    return os.path.abspath(__file__)


def get_abs_pwd_path():
    """Return the absolute path of the present working directory."""
    return os.path.dirname(get_abs_current_file_path())


def get_abs_module_path():
    """Return the absolute path of the module directory."""
    return os.path.dirname(get_abs_pwd_path())


def get_abs_project_path():
    """Return the absolute path of the module directory."""
    return os.path.dirname(get_abs_module_path())
