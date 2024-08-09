import os


class BaseDatasetPathManager:
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
        self.dataset_name = dataset.dn
        self.dataset_type = dataset.dataset_type
        self.data_dp = os.path.join(data_dp, self.dataset_name)
        self.images_dp = os.path.join(
            images_dp, self.dataset_name, tile_str
        )
        self.images_cover_csv_fp = os.path.join(
            self.images_dp, images_cover_csv_fn
        )
        self.labels_dp = os.path.join(
            labels_dp, self.dataset_name, tile_str
        )
        if panoptic_labels_dp is not None:
            self.panoptic_labels_dp = os.path.join(
                panoptic_labels_dp, self.dataset_name, tile_str
            )
        else:
            self.panoptic_labels_dp = None
