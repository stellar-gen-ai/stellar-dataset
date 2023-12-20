# STELLAR Dataset


Code for our paper:
[Stellar: Systematic Evaluation of Human-Centric Personalized Text-to-Image Methods](https://stellar-gen-ai.github.io/#authors)

Authors:
[Panos Achlioptas](https://optas.github.io/), [Alexandros Benetatos](), [Iordanis Fostiropoulos](https://iordanis.me), [Dimitris Skourtis]()

This folder containts information on how to download and setup the STELLAR dataset.

The codebase is maintained by [Alexandros Benetatos](). For any questions please reach out.


## License

Before downloading or using any part of the code in this repository, please review and acknowledge the terms and conditions set forth in both the ["License Terms"](./LICENSE) and ["Third Party License Terms"](./THIRD-PARTIES-LICENSE) included in this repository. Continuing to download and use any part of the code in this repository confirms you agree with these terms and conditions.

## Usage

Summary:

1. Process CelebAMask-HQ Dataset
2. Process STELLAR Prompts
3. Run

### 1. Process CelebAMask-HQ

Assuming you have access to CelebAMask-HQ, you can extract the images we use for our experiments with Stellar, by running the following command:

```bash
python scripts/extract_stellar_from_celebamaskhq.py --dataset-dir output_dir/STELLAR/
```

Where `output_dir/STELLAR/` is the path to the directory where you downloaded CelebAMask-HQ. This script will extract the images and masks from CelebAMask-HQ and place them in the `output_dir/STELLAR/` directory.

After extracting the STELLAR images, the directory structure should be:

    .
    ├── ...
    ├── output_dir                           # The datasets folder
    │   ├── STELLAR                          # The STELLAR folder
    │   │   ├── 000                          # Zeroth subject folder
    │   │   │   ├── 0.jpg                    # First image file
    │   │   │   ├── 0_bg.png                 # First image mask file
    │   │   │   ├── 0_attributes.json        # First image celeba annotations
    │   │   │   ├── 1.jpg                    # Second image file
    │   │   │   ├── 1_bg.png                 # Second image mask file
    │   │   │   ├── 1_attributes.json        # Second image celeba annotations
    │   │   │ ...
    └── ...

### 2. Process Prompts

To download the prompts you need to first accept the terms of use for Stellar Prompts Dataset [here](https://forms.gle/efUfbSqbn9rH77mo8). You will then be provided with a download URL.

To see in details how we build our prompt dataset and associated to the underlying human identities, please visit [our suplemental](https://stellar-gen-ai.github.io/materials/stellar_supplementary.pdf).

After downloading, place `objects.txt` and `prompts.json` in the `./prompts/` directory.

**Place STELLAR Prompts With The Dataset**

After completing the above steps, you would need to run the `place_prompts_with_dataset.py` script from the `scripts` folder to add the prompts for each STELLAR subjects identity in the corresponding folder:

```bash
python scripts/place_prompts_with_dataset.py --dataset-dir output_dir/STELLAR/
```

Where `output_dir/STELLAR/` is the path to the directory where the rest of the dataset is.

After doing all the above steps, the final directory structure for STELLAR should be:

    .
    ├── ...
    ├── output_dir                           # The datasets folder
    │   ├── STELLAR                          # The STELLAR folder
    │   │   ├── 000                          # Zeroth subject folder
    │   │   │   ├── 0.jpg                    # First image file
    │   │   │   ├── 0_bg.png                 # First image mask file
    │   │   │   ├── 0_attributes.json        # First image celeba annotations
    │   │   │   ├── 1.jpg                    # Second image file
    │   │   │   ├── 1_bg.png                 # Second image mask file
    │   │   │   ├── 1_attributes.json        # Second image celeba annotations
    │   │   │   ├── prompts.json             # Prompts file
    │   │   │ ...
    └── ...

## 3. Run

To install the dataset class

```bash
pip install git+https://github.com/stellar-gen-ai/stellar-dataset.git
```

To use the dataset you can simply run:

```python
from stellar_dataset import Stellar

dataset = Stellar(dataset_dir)
```
Where `dataset_dir` is the directory of `STELLAR` dataset from **Download Instructions**

## Dataset Statistics Summary

| Annotation | Unique |
|-----------------|--------|
| Prompt         | 10000  |
| Detectables     | 350    |

| Statistics         | Average | Maximum | Minimum |
|--------------------|---------|---------|---------|
| Tokens/Prompt      | 7.1     | 16      | 2       |
| Detectables/Prompt | 1.5     | 3       | 0       |

