import os
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor
from cllm_data_curation.thestack_curation.general_utils import get_optimal_worker_count


def git_lfs_check(install_style="brew"):
    """ Check if Git LFS is installed and if not then install with indicated method

    Args:
        install_style (str, optional): the method to use to install Git LFS (default for MacOS)

    Returns:
        None; installs Git LFS if not already installed
    """
    try:
        subprocess.check_output(['git', 'lfs', 'version'])
    except subprocess.CalledProcessError:
        print(f'\n... Git LFS is not installed – Launching install for {install_style} ...\n')

        # Install Git LFS using Homebrew on macOS
        if "brew" in install_style:
            subprocess.call(
                ['/usr/bin/ruby', '-e',
                 '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)'])
            subprocess.call(['brew', 'install', 'git-lfs'])
        # Install Git LFS using apt-get on Ubuntu or Debian
        else:
            # Install Git LFS using apt-get on Ubuntu or Debian
            subprocess.call(['sudo', 'apt-get', 'install', 'git-lfs'])
    else:
        print('\n... Git LFS is already installed ...\n')


def clone_git_repo(output_dir, num_workers=None, stack_version="the-stack-dedup",
                   stack_git_url_root="https://huggingface.co/datasets/bigcode"):
    """Clone the Git repository at the specified URL into the specified output directory.

    Args:
        output_dir (str): the path to the directory to clone the Git repository into
        num_workers (int, optional): the number of workers to use when cloning the Git repository
        stack_version (str, optional): the version of the Stack dataset to clone
        stack_git_url_root (str, optional): the root URL for the Stack dataset Git repository

    Returns:
        None; clones the Git repository into the specified output directory
    """
    # Determine the optimal number of workers to use if not provided
    if num_workers is None:
        num_workers = get_optimal_worker_count()

    # Construct the URL for the Git repository
    stack_git_url = os.path.join(stack_git_url_root, stack_version)

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    # Construct the Git clone command
    cmd = ['git', 'clone']

    # Add the number of workers to the command if more than one worker is desired
    if num_workers > 1:
        cmd += ['-j', str(num_workers)]

    # Add the Git repository URL and output directory to the command
    cmd += [stack_git_url, output_dir]

    # Clone the Git repository using subprocess
    subprocess.call(cmd)


def requests_download(url, root_output_dir, auth_token=None):
    """Download the file at the specified URL into the specified output directory.

    Args:
        url (str): the URL of the file to download
        root_output_dir (str): the path to the directory to download the file into
            --> Note that the file will be downloaded into a directory representing
                the language the file is written in inside the `root_output_dir`
                i.e. '/path/to/output_dir/<language>/data-xxxxx-of-xxxxx.parquet'
        auth_token (str, optional): the authentication token to use to download the file
            if not previously authenticated (should be already hence the default is None)

    Returns:
        None; downloads the file into the specified output directory
    """
    # Get the output directory for the language the file is written in
    _lang_dir = os.path.join(root_output_dir, url.rsplit("/", 2)[-2])

    # Create the output directory if it doesn't exist
    if not os.path.isdir(_lang_dir):
        if not os.path.isdir(root_output_dir):
            os.makedirs(root_output_dir, exist_ok=True)
        os.makedirs(_lang_dir, exist_ok=True)
    output_fpath = os.path.join(_lang_dir, url.rsplit("/", 1)[-1])

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        with open(output_fpath, "wb") as f:
            f.write(r.content)
        print(f"File downloaded: {output_fpath}")
    else:
        print(f"Failed to download the file. Status code: {r.status_code}")


def requests_parallel_download(url_list, root_output_dir, auth_token=None, num_workers=None):
    """ Download the files at the specified URLs into the specified output directory using multiple workers

     Args:
        url_list (list):
            – the list of URLs of the files to download
        root_output_dir (str):
            – the path to the directory to download the files into
                --> Note that the files will be downloaded into a directory representing
                    the language the files are written in inside the `root_output_dir`
                    i.e. '/path/to/output_dir/<language>/data-xxxxx-of-xxxxx.parquet'
        auth_token (str, optional):
            – the authentication token to use to download the files if not previously
              authenticated (should be already hence the default is None)

    Returns:
        None; downloads the files into the specified output directory
     """
    if num_workers is None:
        num_workers = get_optimal_worker_count()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [executor.submit(requests_download, url, root_output_dir, auth_token) for url in url_list]
        for task in tasks:
            task.result()
