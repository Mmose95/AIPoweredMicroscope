"""
View individual tiles from a whole-slide microscopy image.

The requested sample is a 3DHISTECH/Pannoramic MRXS slide, not a plain DICOM
file. This script keeps the requested filename but uses OpenSlide so the slide
can be viewed tile-by-tile without loading the full image into memory.

Default slide:
    C:\\Users\\SH37YE\\Desktop\\MRH - Herlev - Ekspektorater\\113801318541

Examples:
    python viewDICOMTiles.py
    python viewDICOMTiles.py --row 200 --col 100
    python viewDICOMTiles.py --row 200 --col 100 --save --no-show
    python viewDICOMTiles.py --tile-width 512 --tile-height 512 --origin slide
"""

from __future__ import annotations

import argparse
import configparser
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import openslide
    from openslide import OpenSlideError, OpenSlideUnsupportedFormatError
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "OpenSlide is required for MRXS/Pannoramic slides.\n"
        "Install it with:\n"
        "  pip install openslide-python openslide-bin\n"
    ) from exc

from PIL import Image, ImageDraw


DEFAULT_SLIDE = Path(
    r"C:\Users\SH37YE\Desktop\MRH - Herlev - Ekspektorater\113801318541"
)
DEFAULT_OUTPUT_DIR = Path("DICOMTilesOutput")


@dataclass(frozen=True)
class SlidePaths:
    input_path: Path
    slide_path: Path
    data_dir: Optional[Path]


@dataclass(frozen=True)
class TileGeometry:
    origin_x: int
    origin_y: int
    area_width: int
    area_height: int
    tile_width: int
    tile_height: int
    max_col: int
    max_row: int


@dataclass(frozen=True)
class TileSelection:
    level: int
    row: int
    col: int
    base_x: int
    base_y: int
    tile_width: int
    tile_height: int
    downsample: float


def resolve_slide_path(input_path: Path) -> SlidePaths:
    """Resolve either an MRXS file or its data folder to an OpenSlide path."""
    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        candidate = input_path.parent / f"{input_path.name}.mrxs"
        if candidate.exists():
            return SlidePaths(input_path=input_path, slide_path=candidate, data_dir=input_path)

        candidates = sorted(input_path.parent.glob("*.mrxs"))
        if len(candidates) == 1:
            return SlidePaths(input_path=input_path, slide_path=candidates[0], data_dir=input_path)

        raise FileNotFoundError(
            f"Could not find sibling MRXS file for data folder: {input_path}\n"
            f"Expected: {candidate}"
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Slide path does not exist: {input_path}")

    if input_path.suffix.lower() == ".mrxs":
        data_dir = input_path.with_suffix("")
        return SlidePaths(
            input_path=input_path,
            slide_path=input_path,
            data_dir=data_dir if data_dir.exists() else None,
        )

    return SlidePaths(input_path=input_path, slide_path=input_path, data_dir=None)


def read_slidedat_ini(data_dir: Optional[Path]) -> Optional[configparser.ConfigParser]:
    if data_dir is None:
        return None

    ini_path = data_dir / "Slidedat.ini"
    if not ini_path.exists():
        return None

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            parser.read(ini_path, encoding=encoding)
            return parser
        except UnicodeDecodeError:
            parser.clear()

    return None


def get_ini_int(
    ini: Optional[configparser.ConfigParser],
    section: str,
    key: str,
    default: Optional[int] = None,
) -> Optional[int]:
    if ini is None or not ini.has_section(section) or not ini.has_option(section, key):
        return default
    try:
        return int(float(ini.get(section, key)))
    except ValueError:
        return default


def get_ini_float(
    ini: Optional[configparser.ConfigParser],
    section: str,
    key: str,
    default: Optional[float] = None,
) -> Optional[float]:
    if ini is None or not ini.has_section(section) or not ini.has_option(section, key):
        return default
    try:
        return float(ini.get(section, key))
    except ValueError:
        return default


def default_tile_size(
    ini: Optional[configparser.ConfigParser],
    level: int,
    tile_width_arg: Optional[int],
    tile_height_arg: Optional[int],
) -> tuple[int, int]:
    section = f"LAYER_0_LEVEL_{level}_SECTION"
    ini_width = get_ini_int(ini, section, "DIGITIZER_WIDTH")
    ini_height = get_ini_int(ini, section, "DIGITIZER_HEIGHT")

    tile_width = tile_width_arg or ini_width or 512
    tile_height = tile_height_arg or ini_height or 512

    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile width and height must be positive.")

    return tile_width, tile_height


def get_scanned_area(
    ini: Optional[configparser.ConfigParser],
    slide_width: int,
    slide_height: int,
    origin_mode: str,
) -> tuple[int, int, int, int]:
    if origin_mode == "slide":
        return 0, 0, slide_width, slide_height

    section = "NONHIERLAYER_2_LEVEL_0_SECTION"
    left = get_ini_int(
        ini, section, "COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__LEFT"
    )
    top = get_ini_int(
        ini, section, "COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__TOP"
    )
    right = get_ini_int(
        ini, section, "COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__RIGHT"
    )
    bottom = get_ini_int(
        ini, section, "COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__BOTTOM"
    )

    if None in (left, top, right, bottom):
        return 0, 0, slide_width, slide_height

    left = max(0, min(slide_width, int(left)))
    top = max(0, min(slide_height, int(top)))
    right = max(left + 1, min(slide_width, int(right)))
    bottom = max(top + 1, min(slide_height, int(bottom)))
    return left, top, right - left, bottom - top


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_geometry(
    slide: openslide.OpenSlide,
    ini: Optional[configparser.ConfigParser],
    level: int,
    origin_mode: str,
    tile_width: int,
    tile_height: int,
) -> TileGeometry:
    slide_width, slide_height = slide.dimensions
    origin_x, origin_y, area_width, area_height = get_scanned_area(
        ini, slide_width, slide_height, origin_mode
    )

    downsample = float(slide.level_downsamples[level])
    level_area_width = max(1, math.ceil(area_width / downsample))
    level_area_height = max(1, math.ceil(area_height / downsample))
    max_col = max(0, math.ceil(level_area_width / tile_width) - 1)
    max_row = max(0, math.ceil(level_area_height / tile_height) - 1)

    return TileGeometry(
        origin_x=origin_x,
        origin_y=origin_y,
        area_width=area_width,
        area_height=area_height,
        tile_width=tile_width,
        tile_height=tile_height,
        max_col=max_col,
        max_row=max_row,
    )


def select_tile(
    slide: openslide.OpenSlide,
    geometry: TileGeometry,
    level: int,
    row: int,
    col: int,
) -> TileSelection:
    downsample = float(slide.level_downsamples[level])
    row = clamp(row, 0, geometry.max_row)
    col = clamp(col, 0, geometry.max_col)
    base_x = int(round(geometry.origin_x + col * geometry.tile_width * downsample))
    base_y = int(round(geometry.origin_y + row * geometry.tile_height * downsample))

    return TileSelection(
        level=level,
        row=row,
        col=col,
        base_x=base_x,
        base_y=base_y,
        tile_width=geometry.tile_width,
        tile_height=geometry.tile_height,
        downsample=downsample,
    )


def rgba_to_rgb_on_white(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.getchannel("A"))
    return background


def read_tile(slide: openslide.OpenSlide, selection: TileSelection) -> Image.Image:
    tile = slide.read_region(
        (selection.base_x, selection.base_y),
        selection.level,
        (selection.tile_width, selection.tile_height),
    )
    return rgba_to_rgb_on_white(tile)


def save_tile_image(
    tile: Image.Image,
    selection: TileSelection,
    slide_path: Path,
    output: Optional[str],
    output_dir: Path,
) -> Path:
    if output:
        output_path = Path(output)
        if output_path.suffix:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(output_path)
            return output_path

        output_dir = output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{slide_path.stem}_level{selection.level}"
        f"_row{selection.row:04d}_col{selection.col:04d}"
        f"_x{selection.base_x}_y{selection.base_y}.png"
    )
    output_path = output_dir / filename
    tile.save(output_path)
    return output_path


def make_overview_base(slide: openslide.OpenSlide) -> tuple[Image.Image, int, float]:
    overview_level = slide.level_count - 1
    overview_size = slide.level_dimensions[overview_level]
    overview = slide.read_region((0, 0), overview_level, overview_size)
    return rgba_to_rgb_on_white(overview), overview_level, float(slide.level_downsamples[overview_level])


def draw_overview_marker(
    overview_base: Image.Image,
    overview_downsample: float,
    selection: TileSelection,
) -> Image.Image:
    overview = overview_base.copy()
    draw = ImageDraw.Draw(overview)

    x0 = selection.base_x / overview_downsample
    y0 = selection.base_y / overview_downsample
    x1 = (selection.base_x + selection.tile_width * selection.downsample) / overview_downsample
    y1 = (selection.base_y + selection.tile_height * selection.downsample) / overview_downsample

    # Draw multiple rectangles so the marker stays visible on brightfield images.
    for offset, color in ((0, "black"), (2, "red"), (4, "white")):
        draw.rectangle(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            outline=color,
            width=2,
        )

    return overview


class InteractiveTileViewer:
    def __init__(
        self,
        slide: openslide.OpenSlide,
        slide_path: Path,
        level: int,
        row: int,
        col: int,
        ini: Optional[configparser.ConfigParser],
        origin_mode: str,
        tile_width_arg: Optional[int],
        tile_height_arg: Optional[int],
        output_dir: Path,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.slide = slide
        self.slide_path = slide_path
        self.level = level
        self.row = row
        self.col = col
        self.ini = ini
        self.origin_mode = origin_mode
        self.tile_width_arg = tile_width_arg
        self.tile_height_arg = tile_height_arg
        self.output_dir = output_dir
        self.overview_base, self.overview_level, self.overview_downsample = make_overview_base(slide)

        self.fig, (self.ax_overview, self.ax_tile) = plt.subplots(1, 2, figsize=(13, 7))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.selection: Optional[TileSelection] = None
        self.geometry: Optional[TileGeometry] = None
        self.tile_image_artist = None
        self.overview_image_artist = None

    def current_geometry(self) -> TileGeometry:
        tile_width, tile_height = default_tile_size(
            self.ini, self.level, self.tile_width_arg, self.tile_height_arg
        )
        return make_geometry(
            self.slide, self.ini, self.level, self.origin_mode, tile_width, tile_height
        )

    def update(self) -> None:
        self.geometry = self.current_geometry()
        self.selection = select_tile(self.slide, self.geometry, self.level, self.row, self.col)
        self.row = self.selection.row
        self.col = self.selection.col

        tile = read_tile(self.slide, self.selection)
        overview = draw_overview_marker(
            self.overview_base, self.overview_downsample, self.selection
        )

        if self.tile_image_artist is None:
            self.overview_image_artist = self.ax_overview.imshow(overview)
            self.tile_image_artist = self.ax_tile.imshow(tile)
            self.ax_overview.axis("off")
            self.ax_tile.axis("off")
        else:
            self.overview_image_artist.set_data(overview)
            self.tile_image_artist.set_data(tile)

        self.ax_overview.set_title(
            f"Overview level {self.overview_level} | red box = selected tile"
        )
        self.ax_tile.set_title(
            f"Tile level {self.selection.level}, row {self.selection.row}/{self.geometry.max_row}, "
            f"col {self.selection.col}/{self.geometry.max_col}\n"
            f"Level-0 x={self.selection.base_x}, y={self.selection.base_y}, "
            f"size={self.selection.tile_width}x{self.selection.tile_height}"
        )
        self.fig.suptitle(
            "Arrow keys: move tile | PageUp/PageDown: change level | s: save | q: quit",
            fontsize=10,
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        if event.key == "left":
            self.col -= 1
        elif event.key == "right":
            self.col += 1
        elif event.key == "up":
            self.row -= 1
        elif event.key == "down":
            self.row += 1
        elif event.key == "pageup":
            self.level = clamp(self.level - 1, 0, self.slide.level_count - 1)
        elif event.key == "pagedown":
            self.level = clamp(self.level + 1, 0, self.slide.level_count - 1)
        elif event.key == "home":
            if self.geometry is not None:
                self.row = self.geometry.max_row // 2
                self.col = self.geometry.max_col // 2
        elif event.key == "s":
            if self.selection is not None:
                tile = read_tile(self.slide, self.selection)
                saved = save_tile_image(
                    tile, self.selection, self.slide_path, None, self.output_dir
                )
                print(f"[SAVED] {saved}")
        elif event.key == "q":
            self.plt.close(self.fig)
            return
        else:
            return

        self.update()

    def show(self) -> None:
        self.update()
        self.plt.show()


def print_slide_info(
    slide: openslide.OpenSlide,
    paths: SlidePaths,
    ini: Optional[configparser.ConfigParser],
    geometry: TileGeometry,
    selection: TileSelection,
) -> None:
    vendor = slide.properties.get("openslide.vendor", "unknown")
    objective = slide.properties.get("openslide.objective-power", "unknown")
    mpp_x = slide.properties.get("openslide.mpp-x") or get_ini_float(
        ini, "LAYER_0_LEVEL_0_SECTION", "MICROMETER_PER_PIXEL_X"
    )
    mpp_y = slide.properties.get("openslide.mpp-y") or get_ini_float(
        ini, "LAYER_0_LEVEL_0_SECTION", "MICROMETER_PER_PIXEL_Y"
    )

    print(f"[INFO] Input: {paths.input_path}")
    print(f"[INFO] OpenSlide file: {paths.slide_path}")
    if paths.data_dir is not None:
        print(f"[INFO] Data folder: {paths.data_dir}")
    print(f"[INFO] Vendor: {vendor}, objective: {objective}x, MPP: {mpp_x}, {mpp_y}")
    print(f"[INFO] Levels: {slide.level_count}")
    for idx, (dims, downsample) in enumerate(
        zip(slide.level_dimensions, slide.level_downsamples)
    ):
        print(f"       level {idx}: {dims[0]} x {dims[1]}, downsample {downsample:g}")
    print(
        f"[INFO] Tile origin: ({geometry.origin_x}, {geometry.origin_y}), "
        f"area: {geometry.area_width} x {geometry.area_height}"
    )
    print(
        f"[INFO] Selected tile: level {selection.level}, row {selection.row}/{geometry.max_row}, "
        f"col {selection.col}/{geometry.max_col}, "
        f"level-0 x={selection.base_x}, y={selection.base_y}, "
        f"size={selection.tile_width} x {selection.tile_height}"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "View an individual tile from an OpenSlide-compatible whole-slide image. "
            "For the supplied sample, pass either the 113801318541 folder or "
            "113801318541.mrxs."
        )
    )
    parser.add_argument(
        "--slide",
        type=Path,
        default=DEFAULT_SLIDE,
        help="Path to an .mrxs file or its associated data folder.",
    )
    parser.add_argument("--level", type=int, default=0, help="OpenSlide level to view.")
    parser.add_argument("--row", "--tile-row", type=int, default=None, help="Tile row.")
    parser.add_argument("--col", "--tile-col", type=int, default=None, help="Tile column.")
    parser.add_argument(
        "--x",
        type=int,
        default=None,
        help="Level-0 x coordinate. If set with --y, overrides row/col selection.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="Level-0 y coordinate. If set with --x, overrides row/col selection.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help="Tile width at the selected level. Defaults to Slidedat.ini DIGITIZER_WIDTH.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Tile height at the selected level. Defaults to Slidedat.ini DIGITIZER_HEIGHT.",
    )
    parser.add_argument(
        "--origin",
        choices=("scanned-area", "slide"),
        default="scanned-area",
        help=(
            "Tile grid origin. 'scanned-area' starts at the sample scanning bounds "
            "from Slidedat.ini; 'slide' starts at the full OpenSlide canvas origin."
        ),
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save the selected tile. Optional value can be a PNG filename or output "
            "directory. With no value, saves to --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory used by --save without filename and by interactive 's'.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib viewer. Useful with --save.",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Print slide/tile metadata and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_slide_path(args.slide)

    try:
        slide = openslide.OpenSlide(str(paths.slide_path))
    except OpenSlideUnsupportedFormatError:
        print(
            "[ERROR] This file is not supported by OpenSlide. "
            "The provided Herlev sample is MRXS/Pannoramic and should be opened by "
            "passing the folder or the .mrxs file. True DICOM WSI files require a "
            "separate pydicom/large-image workflow.",
            file=sys.stderr,
        )
        return 2
    except OpenSlideError as exc:
        print(f"[ERROR] Failed to open slide: {exc}", file=sys.stderr)
        return 2

    with slide:
        if args.level < 0 or args.level >= slide.level_count:
            print(
                f"[ERROR] --level must be between 0 and {slide.level_count - 1}.",
                file=sys.stderr,
            )
            return 2

        ini = read_slidedat_ini(paths.data_dir)
        tile_width, tile_height = default_tile_size(
            ini, args.level, args.tile_width, args.tile_height
        )
        geometry = make_geometry(
            slide, ini, args.level, args.origin, tile_width, tile_height
        )

        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                print("[ERROR] Use both --x and --y, or neither.", file=sys.stderr)
                return 2
            downsample = float(slide.level_downsamples[args.level])
            col = int((args.x - geometry.origin_x) / (tile_width * downsample))
            row = int((args.y - geometry.origin_y) / (tile_height * downsample))
        else:
            row = args.row if args.row is not None else geometry.max_row // 2
            col = args.col if args.col is not None else geometry.max_col // 2

        selection = select_tile(slide, geometry, args.level, row, col)
        print_slide_info(slide, paths, ini, geometry, selection)

        if args.save is not None:
            tile = read_tile(slide, selection)
            save_target = args.save if args.save else None
            saved = save_tile_image(tile, selection, paths.slide_path, save_target, args.output_dir)
            print(f"[SAVED] {saved}")

        if args.info_only or args.no_show:
            return 0

        viewer = InteractiveTileViewer(
            slide=slide,
            slide_path=paths.slide_path,
            level=args.level,
            row=selection.row,
            col=selection.col,
            ini=ini,
            origin_mode=args.origin,
            tile_width_arg=args.tile_width,
            tile_height_arg=args.tile_height,
            output_dir=args.output_dir,
        )
        viewer.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
