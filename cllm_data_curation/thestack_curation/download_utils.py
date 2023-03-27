import os
import sys
import subprocess


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
                ['/usr/bin/ruby', '-e', '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)'])
            subprocess.call(['brew', 'install', 'git-lfs'])
        # Install Git LFS using apt-get on Ubuntu or Debian
        else:
            # Install Git LFS using apt-get on Ubuntu or Debian
            subprocess.call(['sudo', 'apt-get', 'install', 'git-lfs'])
    else:
        print('\n... Git LFS is already installed ...\n')


def clone_git_repo(url, output_dir,
                   num_workers=None, stack_version="the-stack-dedup",
                   stack_git_url_root="https://huggingface.co/datasets/bigcode"):
    """Clone the Git repository at the specified URL into the specified output directory.

    Args:
        url (str): the URL of the Git repository to clone
        output_dir (str): the path to the directory to clone the Git repository into
        num_workers (int, optional): the number of workers to use when cloning the Git repository
        stack_version (str, optional): the version of the Stack dataset to clone
        stack_git_url_root (str, optional): the root URL for the Stack dataset Git repository

    Returns:
        None; clones the Git repository into the specified output directory
    """
    stack_git_url = os.path.join(stack_git_url_root, stack_version)

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    # Construct the Git clone command with the desired number of workers
    cmd = ['git', 'clone']
    if num_workers > 1:
        cmd += ['-j', str(num_workers)]
    cmd += [stack_git_url, output_dir]

    # Clone the Git repository using subprocess
    subprocess.call(cmd)


def download_thestack(output_dir, stack_version="the-stack-dedup", method="git_lfs"):
    """ Download the specified version of the Stack dataset

    Args:
        output_dir (str): the path to the directory to clone the Git repository into

    """

    if stack_version not in ["the-stack-dedup", "the-stack"]:
        raise NotImplementedError(f"stack_version={stack_version} not implemented")

    if method=="git_lfs":
        git_lfs_check()

    elif method=="wget":
        pass
    elif method=="hf":
        pass
    else:
        raise NotImplementedError(f"method={method} not implemented")

