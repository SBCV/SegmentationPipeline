from pydantic import BaseModel
from typing import List, Tuple, Union
from eot.tiles.tile_alignment import TileAlignment


class DatasetEntry(BaseModel):
    # dn = directory name
    dn: str
    dataset_type: str


class RegexEntry(BaseModel):
    # dn = directory name
    dn: str
    regex: str


class BandEntry(BaseModel):
    # dn = directory name
    dn: str
    bands: List[int]


class DataConfig(BaseModel):
    train_datasets: List[DatasetEntry]
    test_datasets: List[DatasetEntry]
    image_search_regex: List[RegexEntry]
    image_ignore_regex: List[RegexEntry]
    label_search_regex: List[RegexEntry]
    label_ignore_regex: List[RegexEntry]
    image_bands: List[BandEntry]
    label_bands: List[BandEntry]
    workspace_dp: str
    training_category_titles: List[str] = None
    invalid_category_target_index: int = 0


class TileConfig(BaseModel):
    # NB: tile_size = (tile_width, tile_height)
    tiling_scheme: str
    input_tile_zoom_level: int = None
    input_tile_size_in_pixel: Tuple[int, int] = None
    input_tile_size_in_meter: Tuple[float, float] = None
    # Stride values used for training
    input_train_tile_stride_in_pixel: Tuple[int, int] = (
        input_tile_size_in_pixel
    )
    input_train_tile_stride_in_meter: Tuple[float, float] = (
        input_tile_size_in_meter
    )

    input_train_tile_alignment: str = TileAlignment.centered_to_image.value
    input_train_tile_overhang: Union[bool, int] = False
    input_train_tile_keep_border_tiles: Union[bool, int] = False

    input_test_tile_stride_in_pixel: Tuple[int, int] = input_tile_size_in_pixel
    input_test_tile_stride_in_meter: Tuple[float, float] = (
        input_tile_size_in_meter
    )
    input_test_tile_alignment: str = TileAlignment.centered_to_image.value
    input_test_tile_overhang: Union[bool, int] = False
    input_test_tile_keep_border_tiles: Union[bool, int] = False

    # Skip tile if nodata pixel ratio > threshold (i.e. 100 == "keep all")
    no_data_threshold: int = 100