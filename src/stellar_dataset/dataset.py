"""Pytorch Dataset used to load the Stellar dataset."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove
from torch.utils.data import Dataset

image_ext = {".png", ".jpg", ".jpeg"}


class Stellar(Dataset):
    def __init__(
        self,
        data_root: Path,
        sample_transform=None,
        split: str = "test",  # "val" | "test"
    ) -> None:
        identity_folders = sorted(
            Path(data_root).glob("*"), key=lambda x: x.stem
        )
        self.data_root = data_root
        self.image_paths: list[Path] = []
        self.metadata: list[dict] = []
        self.prompts: list[list[str]] = []
        self.is_stellar_t: list[list[bool]] = []
        self.detectables: list[list[list[str]]] = []
        self.categories: list[list[list[str]]] = []

        self.sample_transform = sample_transform
        for identity in identity_folders:
            if not identity.is_dir():
                continue
            if split == "val" and int(identity.stem) < 200:
                continue
            if split == "test" and int(identity.stem) >= 200:
                break

            image_paths = sorted(
                [
                    p
                    for p in identity.glob("*")
                    if p.suffix in image_ext and not p.stem.endswith("_bg")
                ],
                key=lambda x: x.name,
            )
            metadata = [
                json.loads(p.read_text())
                for p in sorted(
                    [
                        p
                        for p in identity.glob("*")
                        if p.suffix in {".json"}
                        and p.stem.endswith("_attributes")
                    ],
                    key=lambda x: x.name,
                )
            ]
            prompts = json.loads((identity / "prompts_t.json").read_text())[
                "prompts"
            ]
            is_stellar_t = [True] * len(prompts)
            detectables = json.loads((identity / "prompts_t.json").read_text())[
                "detectables"
            ]
            categories = json.loads((identity / "prompts_t.json").read_text())[
                "categories"
            ]
            if split == "test":
                prompts_h = json.loads(
                    (identity / "prompts_h.json").read_text()
                )["prompts"]
                is_stellar_t += [False] * len(prompts_h)
                prompts = prompts + prompts_h
                detectables += [[]] * len(prompts_h)
                categories += [[]] * len(prompts_h)

            assert len(image_paths) == len(metadata)
            self.image_paths += image_paths
            self.metadata += metadata

            assert (
                len(detectables)
                == len(prompts)
                == len(categories)
                == len(is_stellar_t)
            )
            self.prompts += prompts
            self.is_stellar_t += is_stellar_t
            self.detectables += detectables
            self.categories += categories

        if len(self.prompts) == 0:
            raise RuntimeError(f"No dataset was found in {data_root}.")
        self._num_prompts = len(prompts)
        self._imgs_per_id = len(image_paths)
        lens = [
            len(self.image_paths) // self._imgs_per_id,
            len(self.metadata) // self._imgs_per_id,
            len(self.prompts) // self._num_prompts,
            len(self.detectables) // self._num_prompts,
            len(self.categories) // self._num_prompts,
            len(self.is_stellar_t) // self._num_prompts,
        ]
        assert all(lens[0] == np.array(lens))

    def __len__(self):
        return len(self.image_paths) * self._num_prompts

    def get_metadata(self, idx):
        idx = int(idx)
        img_idx = idx // self._num_prompts
        prompt_idx = ((img_idx // self._imgs_per_id) * self._num_prompts) + (
            idx % self._num_prompts
        )
        img_path = self.image_paths[img_idx]
        text = self.prompts[prompt_idx]
        is_stellar_t = self.is_stellar_t[prompt_idx]
        subject_name = img_path.stem
        metadata = {
            "subject_name": subject_name,
        }
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = img_path.relative_to(self.data_root)
        # formatted as relative_path, file-stem, abs index
        save_name = "-".join(
            [
                str(rel_path.parent).replace("/", "_"),
                rel_path.stem,
                str(prompt_idx).zfill(2),
            ]
        )
        metadata = {
            "index": idx,
            "generation_datetime": datetime_now,
            "prompt": text,
            "is_stellar_t": is_stellar_t,
            "attributes": self.metadata[img_idx]["attributes"],
            "detectables": self.detectables[prompt_idx],
            "categories": self.categories[prompt_idx],
            "image_path": img_path,
            "save_name": save_name,
        }
        return metadata

    def __getitem__(self, index: int) -> dict[str, int | Image.Image | str]:
        img_idx = index // self._num_prompts
        prompt_idx = ((img_idx // self._imgs_per_id) * self._num_prompts) + (
            index % self._num_prompts
        )
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert("RGB")
        mask_path: Path = img_path.parent / (img_path.stem + "_bg.png")
        if not mask_path.exists():
            input_image = Path(self.image_paths[img_idx]).read_bytes()
            output = remove(input_image)
            Path(mask_path).write_bytes(output)
            mask = np.array(Image.open(mask_path))
            Image.fromarray(
                (mask[:, :, -1] != 0).astype(np.uint8) * 255
            ).convert("RGB").save(mask_path)
        mask = Image.open(mask_path)

        prompt = self.prompts[prompt_idx]
        if self.sample_transform is not None:
            sample = self.sample_transform(img=img, mask=mask, prompt=prompt)
        else:
            sample = {"img": img, "mask": mask, "prompt": prompt}
        orig_size = np.array(img.size[::-1])
        sample["orig_size"] = orig_size
        sample["index"] = index
        return sample
