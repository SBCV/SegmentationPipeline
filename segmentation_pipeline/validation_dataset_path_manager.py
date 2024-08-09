import os

from segmentation_pipeline.base_dataset_path_manager import BaseDatasetPathManager


class ValidationDatasetPathManager(BaseDatasetPathManager):
    def __init__(
        self,
        dataset,
        tile_str,
        data_dp,
        images_dp,
        labels_dp,
        images_cover_csv_fn,
        panoptic_labels_dp=None,
    ):
        super().__init__(
            dataset,
            tile_str,
            data_dp,
            images_dp,
            labels_dp,
            images_cover_csv_fn,
            panoptic_labels_dp
        )
