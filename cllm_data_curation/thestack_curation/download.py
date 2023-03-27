import os
import datasets
import argparse

from cllm_data_curation.thestack_curation.download_utils import git_lfs_check
from cllm_data_curation.thestack_curation.download_utils import clone_git_repo
from cllm_data_curation.thestack_curation.general_utils import get_abs_pwd_path
from cllm_data_curation.thestack_curation.general_utils import authenticate_hf
from cllm_data_curation.thestack_curation.general_utils import read_csv_urls
from cllm_data_curation.thestack_curation.download_utils import requests_parallel_download


def download_thestack(output_dir, hf_token, stack_version="the-stack-dedup", method="git_lfs"):
    """ Download the specified version of the Stack dataset

    Args:
        output_dir (str): the path to the directory to clone the Git repository into
        hf_token (str): the HuggingFace token to use to authenticate [required]
        stack_version (str, optional): the version of the Stack dataset to clone
        method (str, optional): the method to use to download the Stack dataset
            --> One of ['git_lfs', 'requests', 'huggingface']

    Returns:
        None; downloads the dataset into the specified output directory
              either by cloning the Git repository or downloading the dataset

    """
    # Authenticate with HuggingFace
    authenticate_hf(hf_token=hf_token, auth_git=True if 'git' in method.lower() else False)

    # Check if the specified stack version is implemented
    if stack_version not in ["the-stack-dedup", "the-stack"]:
        raise NotImplementedError(f"stack_version={stack_version} not implemented")

    if method.lower()=="git_lfs":
        git_lfs_check()
        clone_git_repo(output_dir, stack_version=stack_version)
    elif method.lower()=="requests":
        url_list = read_csv_urls(
            os.path.join(get_abs_pwd_path(), "supplementary_data", f"{stack_version.replace('-', '_')}_urls.csv")
        )
        requests_parallel_download(url_list, root_output_dir=output_dir, auth_token=hf_token)
    elif method.lower() in ["hf", "huggingface"]:
        print("\n... THIS ISN'T WORKING AND USUALLY FAILS ...\n")
        datasets.load_dataset(os.path.join("bigcode", stack_version), split="train", cache_dir=output_dir)
    else:
        raise NotImplementedError(f"method={method} not implemented")


def main():

    # Parser for command line arguments
    parser = argparse.ArgumentParser(description="Download the Stack dataset using one of three methods.")

    # Where to download the dataset to (note you should have at least 1TB of free disk space)
    parser.add_argument(
        "--output_dir",
        help="The output directory where the dataset will be downloaded."
    )

    # Your personal HuggingFace token.
    #    --> You will have to go to the dataset of your choice
    #        and accept the terms & conditions before you can
    #        download the dataset.
    parser.add_argument(
        "--hf_token",
        help="The HuggingFace token to use for authentication."
    )

    # Which version of the Stack dataset to download
    parser.add_argument(
        "--stack_version",
        default="the-stack-dedup",
        choices=["the-stack-dedup", "the-stack"],
        help="The version of the Stack dataset to download. "
             "The dedup version is ~1TB on disk while the non-dedup version is ~2.5TB."
             "NOTE: dedup refers to the near-deduplication performed"
             "by the authors of the Stack dataset. This is done using hashing. "
             "The default is the near-deduplicated version.",
    )

    # Which method to use to download the Stack dataset
    parser.add_argument(
        "--method",
        default="git_lfs",
        choices=["git_lfs", "requests", "huggingface"],
        help="The method to use to download the Stack dataset.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Download the Stack dataset
    download_thestack(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        stack_version=args.stack_version,
        method=args.method,
    )


if __name__ == "__main__":
    main()
