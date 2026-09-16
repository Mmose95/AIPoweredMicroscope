"""Manifest-backed unlabeled image dataset for the official DINOv3 trainer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

from dinov3.data.datasets.extended import ExtendedVisionDataset


class MicroscopyManifest(ExtendedVisionDataset):
    """Read microscopy tiles listed in a CSV without using annotations.

    The manifest must contain ``relative_path``. ``root`` is joined to that
    portable path, which lets the same manifest work in WSL and on UCloud.
    """

    def __init__(
        self,
        *,
        manifest: str,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # ExtendedVisionDataset reserves its first positional arguments for
        # decoder classes, so torchvision dataset options must be named.
        super().__init__(
            root=str(root),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.image_root = Path(root).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"Image root not found: {self.image_root}")

        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "relative_path" not in reader.fieldnames:
                raise ValueError("Manifest must contain a 'relative_path' column")
            self._relative_paths = [row["relative_path"].strip() for row in reader]

        if not self._relative_paths:
            raise ValueError(f"Manifest contains no image rows: {self.manifest_path}")
        if any(not path for path in self._relative_paths):
            raise ValueError("Manifest contains an empty relative_path")

    def __len__(self) -> int:
        return len(self._relative_paths)

    def image_path(self, index: int) -> Path:
        relative = Path(self._relative_paths[index])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative_path at row {index + 2}: {relative}")
        return self.image_root / relative

    def get_image_data(self, index: int) -> bytes:
        path = self.image_path(index)
        try:
            return path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"Cannot read manifest image: {path}") from error

    def get_target(self, index: int) -> int:
        # DINOv3 SSL ignores class labels, but its dataset interface expects a target.
        return 0
