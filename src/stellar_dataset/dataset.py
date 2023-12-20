"""
Dataset used for loading identities from Image Folders, Prompts to create an ``IdentityDataset``
from the cross product of the two.
"""

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
    ) -> None:
        identity_folders = sorted(Path(data_root).glob("*"), key=lambda x: x.stem)
        self.data_root = data_root
        self.image_paths: list[Path] = []
        self.prompts: list[str] = []
        self.metadata: list[dict] = []
        self.detectables: list[list[str]] = []

        self.sample_transform = sample_transform
        for identity in identity_folders:
            if not identity.is_dir():
                continue
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
                        if p.suffix in {".json"} and p.stem.endswith("_attributes")
                    ],
                    key=lambda x: x.name,
                )
            ]
            prompts = [
                json.loads((identity / "prompt.json").read_text())["prompts"]
                for _ in image_paths
            ]
            detectables = [
                json.loads((identity / "prompt.json").read_text())["detectables"]
                for _ in image_paths
            ]
            self.image_paths += image_paths
            self.prompts += prompts
            self.detectables += detectables
            self.metadata += metadata
            assert len(detectables[0]) == len(prompts[0])
        if len(self.prompts) == 0:
            raise RuntimeError(f"No dataset was found in {data_root}.")
        self._num_prompts = len(prompts[0])
        lens = [
            len(self.image_paths),
            len(self.detectables),
            len(self.prompts),
            len(self.metadata),
        ]
        assert all(lens[0] == np.array(lens))

    def __len__(self):
        return len(self.image_paths) * self._num_prompts

    def get_metadata(self, idx):
        idx = int(idx)
        img_idx = idx // self._num_prompts
        prompt_idx = idx % self._num_prompts
        img_path = self.image_paths[img_idx]
        text = self.prompts[img_idx][prompt_idx]
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
            "attributes": self.metadata[img_idx]["attributes"],
            "detectables": self.detectables[img_idx][prompt_idx],
            "image_path": img_path,
            "save_name": save_name,
        }
        return metadata

    def __getitem__(self, index: int) -> dict[str, int | Image.Image | str]:
        img_idx = index // self._num_prompts
        prompt_idx = index % self._num_prompts
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert("RGB")
        mask_path: Path = img_path.parent / (img_path.stem + "_bg.png")
        if not mask_path.exists():
            input_image = Path(self.image_paths[img_idx]).read_bytes()
            output = remove(input_image)
            Path(mask_path).write_bytes(output)
            mask = np.array(Image.open(mask_path))
            Image.fromarray((mask[:, :, -1] != 0).astype(np.uint8) * 255).convert(
                "RGB"
            ).save(mask_path)
        mask = Image.open(mask_path)
        prompt = self.prompts[img_idx][prompt_idx]
        if self.sample_transform is not None:
            sample = self.sample_transform(img=img, mask=mask, prompt=prompt)
        else:
            sample = {"img": img, "mask": mask, "prompt": prompt}
        orig_size = np.array(img.size[::-1])
        sample["orig_size"] = orig_size
        sample["index"] = index
        return sample
