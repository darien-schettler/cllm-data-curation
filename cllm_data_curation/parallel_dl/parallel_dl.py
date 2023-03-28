import argparse

from cllm_data_curation.parallel_dl.multiprocessing_utils import do_work
from cllm_data_curation.parallel_dl.preprocessing_utils import get_repos
from cllm_data_curation.parallel_dl.preprocessing_utils import get_parallel_params
from cllm_data_curation.parallel_dl.preprocessing_utils import get_repo_chunks
from cllm_data_curation.parallel_dl.preprocessing_utils import print_check
from cllm_data_curation.parallel_dl.preprocessing_utils import get_bad_extensions


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel collection of repo data/text")
    parser.add_argument("--segment_number", type=int, default=1,
                        help="Which segment of the repo list to process.")
    parser.add_argument("--frac", type=float, default=0.25,
                        help="Fraction of the repo list segments are based off of.")
    args = parser.parse_args()
    return args


def main(args):
    """ Main function for the parallel collection of repo data/text

    Args:
        args: Command line arguments

    Returns:
        None; writes to an archive.

    """
    print("\n... GETTING REPOLIST ...")
    repo_list = get_repos(frac=args.frac, seg_num=args.segment_number)

    print("... GETTING PARALLEL PARAMS ...")
    parallel_params = get_parallel_params()

    print("... GETTING REPO CHUNKS ...")
    repo_chunks = get_repo_chunks(repo_list, parallel_params['chunk_size'])

    print("... PRINTING CHECKS ...")
    print_check(repo_list=repo_list, repo_chunks=repo_chunks, n_rejects=len(get_bad_extensions()), **parallel_params)

    print("... DO WORK! ...\n")
    do_work(repo_chunks, parallel_params["n_threads"])


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
