import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from os.path import exists as pexists
from os.path import join as pjoin


def parse_dataset(dataset_dir):
    # Check if dataset_dir exists and contains the required files
    if not pexists(dataset_dir):
        raise ValueError(
            f"Invalid dataset directory: {dataset_dir}. "
            "The directory must already exist and contain the required files (see README for more info)."  # noqa
        )

    # Unzip images and masks
    if pexists(pjoin(dataset_dir, "image.zip")):
        for zip_filepath, unzip_dir in zip(
            [
                pjoin(dataset_dir, "image.zip"),
                pjoin(
                    dataset_dir, "mask", "CelebAMask-HQ-mask-color-palette.zip"
                ),
            ],
            [
                pjoin(dataset_dir, "images_r"),
                pjoin(dataset_dir, "masks_r", "colorized_r"),
            ],
        ):
            print(f"Unzipping {zip_filepath} to {unzip_dir} ...")
            if not pexists(unzip_dir):
                with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                    zip_ref.extractall(unzip_dir)
                os.remove(zip_filepath)
    shutil.rmtree(pjoin(dataset_dir, "mask"), ignore_errors=True)

    # Make annotations directory and move files there
    if pexists(pjoin(dataset_dir, "identity", "identity_CelebA-HQ.txt")):
        os.mkdir(pjoin(dataset_dir, "annotations"))
        os.renames(
            pjoin(dataset_dir, "identity", "identity_CelebA-HQ.txt"),
            pjoin(dataset_dir, "annotations", "identity.txt"),
        )
        os.renames(
            pjoin(
                dataset_dir,
                "classification_label",
                "CelebAMask-HQ-attribute-anno.txt",
            ),
            pjoin(dataset_dir, "annotations", "attributes.txt"),
        )
        os.renames(
            pjoin(
                dataset_dir,
                "classification_label",
                "combined_annotation_hq.txt",
            ),
            pjoin(dataset_dir, "annotations", "attributes_finegrained.txt"),
        )

        # Rename and move directories around to make sense
        os.renames(
            pjoin(dataset_dir, "images_r", "image"),
            pjoin(dataset_dir, "images"),
        )
        os.renames(
            pjoin(
                dataset_dir,
                "masks_r",
                "colorized_r",
                "mnt",
                "lustre",
                "share",
                "zqhuang",
                "datasets_face",
                "Face-Diffusion-raw-datasets",
                "CelebA-Dialog-combined",
                "mask",
                "CelebAMask-HQ-mask-color-palette",
            ),
            pjoin(dataset_dir, "masks"),
        )
    if pexists(pjoin(dataset_dir, "text")):
        shutil.rmtree(pjoin(dataset_dir, "text"))

    # Get CelebA-HQ filenames for STELLAR
    with open(
        pjoin(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "images",
            "celebahq_images.txt",
        ),
        "r",
    ) as fp:
        stellar_celebahq_idxs = list(
            map(
                lambda ln: [
                    int(ln.split(",")[0].split(".")[0]),
                    ln.split(",")[1],
                ],
                fp.read().strip("\n").split("\n")[1:],
            )
        )
    stellar_celebahq_idxs = {
        itm[0]: itm[1] for itm in stellar_celebahq_idxs
    }  # {0: 'val', 1: 'test', ...}

    if not pexists(pjoin(dataset_dir, "annotations")):
        exit()

    # Load attributes and keep only those in STELLAR
    with open(pjoin(dataset_dir, "annotations", "attributes.txt"), "r") as fp:
        attributes = fp.read().strip("\n").split("\n")[1:]
    attributes = list(map(lambda ll: ll.strip().split(), attributes))
    attributes_labels = attributes[0]
    attributes = {
        int(itm[0][:-4]): itm[1:]
        for itm in attributes[1:]
        if int(itm[0][:-4]) in stellar_celebahq_idxs.keys()
    }

    # Load finegrained attributes and keep only those in STELLAR
    with open(
        pjoin(dataset_dir, "annotations", "attributes_finegrained.txt"), "r"
    ) as fp:
        finegrained_attributes = fp.read().strip("\n").split("\n")
    finegrained_attributes = list(
        map(lambda ll: ll.strip("\t").split("\t"), finegrained_attributes)
    )
    finegrained_attributes_labels = finegrained_attributes[0][1:]
    # image 5380 does not have a finegrained annotation
    finegrained_attributes = {
        int(fa[0][:-4]): fa[1:]
        for fa in finegrained_attributes[1:]
        if int(fa[0][:-4]) in stellar_celebahq_idxs.keys()
    }

    # Load identities and keep only those in STELLAR
    with open(pjoin(dataset_dir, "annotations", "identity.txt"), "r") as fp:
        identities = fp.read().strip("\n").split("\n")
    identities = list(map(lambda ll: ll.strip().split(), identities))

    identities = {
        int(fa[0][:-4]): int(fa[1])
        for fa in identities
        if int(fa[0][:-4]) in stellar_celebahq_idxs.keys()
    }

    # Invert identities
    identities_inv = defaultdict(list)
    for k, v in identities.items():
        identities_inv[v].append(k)

    identities_inv = {
        idd: [
            celeba_filenames,
            stellar_celebahq_idxs[celeba_filenames[0]],
        ]
        for idd, celeba_filenames in identities_inv.items()
    }
    identities_inv = dict(
        sorted(identities_inv.items(), key=lambda x: (x[1][1], x[0]))
    )

    # Save STELLAR
    for new_id, (old_id, (celebahq_filenames, split)) in enumerate(
        identities_inv.items()
    ):
        save_dir = pjoin(dataset_dir, f"{new_id:03d}")
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass

        sample_annos = {}
        for idx, filename in enumerate(celebahq_filenames):
            os.renames(
                pjoin(dataset_dir, "images", f"{filename}.jpg"),
                pjoin(save_dir, f"{idx}.jpg"),
            )
            os.renames(
                pjoin(dataset_dir, "masks", f"{filename}.png"),
                pjoin(save_dir, f"{idx}_bg.png"),
            )
            sample_annos[idx] = {
                "attributes": {
                    attr_nm: int(attr)
                    for attr_nm, attr in zip(
                        attributes_labels,
                        attributes[filename],
                    )
                },
                "finegrained_attributes": {
                    attr_nm: int(attr)
                    for attr_nm, attr in zip(
                        finegrained_attributes_labels,
                        finegrained_attributes[filename],
                    )
                },
                "identity": old_id,
                "original_filename": f"{filename}.jpg",
                # "split": split,
            }
        for idx, celeba_annos in sample_annos.items():
            with open(pjoin(save_dir, f"{idx}_attributes.json"), "w") as fp:
                json.dump(celeba_annos, fp, indent=4)

    # Delete unnecessary folders
    print("Deleting unnecessary folders (this might take a while) ...")
    if pexists(pjoin(dataset_dir, "annotations")):
        shutil.rmtree(pjoin(dataset_dir, "annotations"))
    if pexists(pjoin(dataset_dir, "images")):
        shutil.rmtree(pjoin(dataset_dir, "images"))
    if pexists(pjoin(dataset_dir, "masks")):
        shutil.rmtree(pjoin(dataset_dir, "masks"))
    if pexists(pjoin(dataset_dir, "mask")):
        shutil.rmtree(pjoin(dataset_dir, "mask"))
    os.remove(pjoin(dataset_dir, "CelebA-HQ-to-CelebA-mapping.txt"))


if __name__ == "__main__":
    # Read a single command line argument of --dataset_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory where the dataset will be saved",
    )
    dataset_dir = parser.parse_args().dataset_dir
    parse_dataset(dataset_dir)
    parse_dataset(dataset_dir)
