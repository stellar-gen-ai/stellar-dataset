import argparse
import json
import os
import shutil
from os.path import exists as pexists
from os.path import join as pjoin


def process(dataset_dir):
    # Check if dataset_dir exists and contains the required files
    if not pexists(dataset_dir):
        raise ValueError(
            f"Invalid dataset directory: {dataset_dir}. "
            "The directory must already exist and contain the required files (see README for more info)."  # noqa
        )

    # Load json with prompts
    with open(
        pjoin(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "prompts.json",
        )
    ) as fp:
        prompts = json.load(fp)

    for subject_folder in filter(lambda nm: nm.isnumeric(), os.listdir(dataset_dir)):
        with open(pjoin(dataset_dir, subject_folder, "prompt.json"), "w") as fp:
            json.dump(prompts[subject_folder], fp, indent=4)

    shutil.copyfile(
        pjoin(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "objects.txt",
        ),
        pjoin(dataset_dir, "objects.txt"),
    )


if __name__ == "__main__":
    # Read a single command line argument of --dataset_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory where the dataset will be saved",
    )
    dataset_dir = parser.parse_args().dataset_dir
    process(dataset_dir)
