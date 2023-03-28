import os
import argparse

from cllm_data_curation.thestack_curation.curation_utils import make_meta_df
from cllm_data_curation.thestack_curation.curation_utils import filter_parquet_file
from cllm_data_curation.thestack_curation.curation_utils import filter_meta_languages
from cllm_data_curation.thestack_curation.curation_configs import ModerateFilterConfig
from cllm_data_curation.thestack_curation.curation_configs import PermissiveFilterConfig
from cllm_data_curation.thestack_curation.curation_configs import AggressiveFilterConfig


def main(_args):
    # Get config based on the given style
    config = None
    if _args.config_style.lower() == "permissive":
        config = PermissiveFilterConfig()
    elif _args.config_style.lower() == "moderate":
        config = ModerateFilterConfig()
    elif _args.config_style.lower() == "aggressive":
        config = AggressiveFilterConfig()
    else:
        raise ValueError(f"Invalid config_style '{_args.config_style}'."
                         f"Accepted styles are one of ['permissive', 'moderate', or 'aggressive'].")

    # Create metadata DataFrame
    print("\n... CREATING INITIAL META DATAFRAME ...")
    meta_df = make_meta_df(_args.root_dir)

    # Filter metadata DataFrame based on the given arguments
    print("... FILTERING UNWANTED LANGUAGES BASED ON PROVIDED ARGUMENTS ...")
    filtered_meta_df = filter_meta_languages(
        meta_df, top_k=_args.top_k, mb_size_thresh=_args.mb_size_thresh, pq_file_cnt_thresh=_args.pq_file_cnt_thresh
    )

    # Iterate over the filtered Parquet files and apply the filtering function
    print("... FILTERING OUT BAD FILES AT PROVIDED CONFIGURATION LEVEL ...")
    filtered_meta_df["filtered_pq_path"] = filtered_meta_df["pq_path"].progress_apply(
        lambda x: filter_parquet_file(x, _args.output_dir, is_slim=_args.is_slim, **config.__dict__)
    )

    print(f"... SAVING FILTERED METADATA TO {args.output_dir} ...\n")
    filtered_meta_df.to_csv(os.path.join(_args.output_dir, "filtered_meta.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate dataset based on the provided configurations and arguments.")
    parser.add_argument("root_dir",
                        help="The directory containing the Parquet files for the-stack or the-stack-dedup dataset.")
    parser.add_argument("output_dir", help="The output directory for the filtered curated Parquet files.")
    parser.add_argument("--config_style", default="permissive",
                        help="The filtering config style (permissive, moderate, or aggressive).")
    parser.add_argument("--top_k", type=int, default=None,
                        help="The top K languages to keep based on size. "
                             "This takes precedence over `the mb_size_thresh` argument.")
    parser.add_argument("--mb_size_thresh", type=float, default=None,
                        help="The minimum size of the language in megabytes. "
                             "This takes precedence over `the pq_file_cnt_thresh` argument`.")
    parser.add_argument("--pq_file_cnt_thresh", type=int, default=None,
                        help="The minimum number of Parquet files in the language.")
    parser.add_argument("--is_slim", action="store_true",
                        help="Whether the Parquet file is a slim version of the full dataset."
                             "If this is the first time you're curating the dataset, you should not use this flag.")

    args = parser.parse_args()
    main(args)
