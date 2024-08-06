import os


class TrainDatasetPathManager:
    def __init__(
        self,
        dataset,
        tile_str,
        train_data_dp,
        train_images_dp,
        train_labels_dp,
        images_cover_csv_fn,
    ):
        self.dataset_name = dataset.dn
        self.dataset_type = dataset.dataset_type
        self.train_data_dp = os.path.join(train_data_dp, self.dataset_name)
        self.train_images_dp = os.path.join(
            train_images_dp, self.dataset_name, tile_str
        )
        self.train_images_cover_csv_fp = os.path.join(
            self.train_images_dp, images_cover_csv_fn
        )
        self.train_labels_dp = os.path.join(
            train_labels_dp, self.dataset_name, tile_str
        )
