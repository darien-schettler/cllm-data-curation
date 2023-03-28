from cllm.processing_utils import process_repo_list
import lm_dataformat as lmd
from tqdm import tqdm
from multiprocessing import Pool
import subprocess
from itertools import repeat


def do_work(repo_chunks, n_threads, commit_freq=10_000, archive_name='./github_data', clon_tout=600, _tmp_dir=".tmp"):
    """ Processes a list of repos in parallel.

    Args:
        repo_chunks (list): List of lists of repos to process.
        n_threads (int): Number of threads to use.
        commit_freq (int): How often to commit to the archive.
        archive_name (str): Name of the archive.
        proc_tout (int): Timeout for processing a single repo.
        clon_tout (int): Timeout for cloning a single repo.
        _tmp_dir (str): Temporary directory to use.

    Returns:
        None; writes to an archive.
    """
    # Create the initial archive
    ar = lmd.Archive(archive_name)

    # Initialize the pool
    pool = Pool(n_threads)

    # Create the progress bar
    pbar = tqdm(repo_chunks, total=len(repo_chunks))

    success_hist = []
    for count, chunk in enumerate(pbar):
        repos_out = pool.starmap(process_repo_list, zip(chunk, repeat(clon_tout), repeat(_tmp_dir)))

        empty_repo_cnt, non_empty_repo_cnt = 0, 0
        for repo in repos_out:
            if repo is not None:
                non_empty_repo_cnt += 1
                for f in repo: ar.add_data(f[0], meta=f[1])
            else:
                empty_repo_cnt += 1

        # remove any leftover files
        subprocess.Popen(
            f"rm -rfv {_tmp_dir} && mkdir {_tmp_dir}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        # Commit at appropriate intervals
        if count % commit_freq == 0:
            ar.commit()

        # Success stats
        success_hist.append((non_empty_repo_cnt / len(repos_out)) * 100)
        success_rate = sum(success_hist) / len(success_hist)
        pbar.set_postfix({"Success Rate": success_rate})
    ar.commit()