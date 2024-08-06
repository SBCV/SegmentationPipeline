import os


class ValidationDatasetPathManager:
    def __init__(
        self,
        dataset,
        tile_str,
        validation_dp,
        validation_data_dp,
        validation_images_dp,
        validation_labels_dp,
        images_cover_csv_fn,
        validation_panoptic_labels_dp=None,
    ):
        self.dataset_name = dataset.dn
        self.dataset_type = dataset.dataset_type
        self.validation_data_dp = os.path.join(validation_data_dp, self.dataset_name)
        self.validation_data_normalized_dp = os.path.join(
            validation_dp, "data_normalized", self.dataset_name
        )
        self.validation_images_dp = os.path.join(
            validation_images_dp, self.dataset_name, tile_str
        )
        self.validation_images_cover_csv_fp = os.path.join(
            self.validation_images_dp, images_cover_csv_fn
        )
        self.validation_labels_dp = os.path.join(
            validation_labels_dp, self.dataset_name, tile_str
        )
        if validation_panoptic_labels_dp is not None:
            self.validation_panoptic_labels_dp = os.path.join(
                validation_panoptic_labels_dp, self.dataset_name, tile_str
            )
        else:
            self.validation_panoptic_labels_dp = None
