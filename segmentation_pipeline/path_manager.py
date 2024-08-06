import os

from test_dataset_path_manager import TestDatasetPathManager
from train_dataset_path_manager import TrainDatasetPathManager
from neat_eo.utility.os_extension import get_subdirs


class PathManager:
    def __init__(
        self,
        workspace_dp,
        requested_train_datasets,
        requested_test_datasets,
        image_search_regex_entries,
        image_ignore_regex_entries,
        label_search_regex_entries,
        label_ignore_regex_entries,
        training_category_titles,
        segmentation_model_cfg_fp,
        segmentation_model_cfg_description_str,
        toml_config_fp,
        train_tiling_scheme,
        test_tiling_scheme,
        validation_tiling_scheme,
        disable_test_pipeline_resizing=None,
        prediction_model_checkpoint_fn="latest.pth",
        prediction_base_tile_prediction=None,
        prediction_base_tile_merging=None,
    ):
        self.workspace_dp = workspace_dp
        self.active_test_dataset_names = None

        assert workspace_dp is not None
        if not os.path.isdir(self.workspace_dp):
            os.mkdir(self.workspace_dp)

        self.segmentation_model_cfg_fp = segmentation_model_cfg_fp
        self.toml_config_fp = toml_config_fp

        # https://stackoverflow.com/questions/20638040/glob-exclude-pattern
        #   In glob the exclude pattern differs from the linux command line
        #   (i.e. "!" vs. "^")
        # For example "[!-]" (glob) or "[^-]" (command line) means that the
        # folder must not end with a "-"
        self.dataset_to_image_search_regex = {
            entry.dn: entry.regex for entry in image_search_regex_entries
        }
        self.dataset_to_image_ignore_regex = {
            entry.dn: entry.regex for entry in image_ignore_regex_entries
        }

        # This can be either *.geojson or *.tif files
        self.dataset_to_label_search_regex = {
            entry.dn: entry.regex for entry in label_search_regex_entries
        }
        self.dataset_to_label_ignore_regex = {
            entry.dn: entry.regex for entry in label_ignore_regex_entries
        }

        self.images_cover_csv_fn = "images_cover.csv"
        self.grid_json_fn = "grid.json"

        # =========================== Training Data ===========================
        self.train_tile_str = self._get_tile_str(train_tiling_scheme)

        self.train_name = "train"
        self.train_dp = os.path.join(self.workspace_dp, self.train_name)
        self.train_data_dp = os.path.join(self.train_dp, "data")
        self.train_images_dp = os.path.join(self.train_dp, "images")
        self.train_labels_dp = os.path.join(self.train_dp, "labels")
        self.train_datasets = self._get_datasets(
            self.train_data_dp, requested_train_datasets
        )

        self.training_category_titles = training_category_titles
        self.mmsegmentation_cfg_description_str = segmentation_model_cfg_description_str
        model_specific_dp_list = self.get_model_specific_dp_list(
            self.training_category_titles,
            self.train_datasets,
            self.train_tile_str,
            self.segmentation_model_cfg_fp,
            self.mmsegmentation_cfg_description_str,
            join_datasets=True
        )
        assert len(model_specific_dp_list) == 1
        self.model_specific_dp = model_specific_dp_list[0]
        self.train_model_dp = self.get_trained_model_dp(
            self.train_dp, self.model_specific_dp
        )
        self.prediction_model_checkpoint_fn = prediction_model_checkpoint_fn
        self.prediction_model_checkpoint_fp = self.get_prediction_model_checkpoint_fp(
            self.train_model_dp, self.prediction_model_checkpoint_fn
        )

        self.train_dataset_path_manager = {
            train_dataset.dn: TrainDatasetPathManager(
                dataset=train_dataset,
                tile_str=self.train_tile_str,
                train_data_dp=self.train_data_dp,
                train_images_dp=self.train_images_dp,
                train_labels_dp=self.train_labels_dp,
                images_cover_csv_fn=self.images_cover_csv_fn,
            )
            for train_dataset in self.train_datasets
        }

        # ========================== Validation Data ==========================
        self.validation_tile_str = self._get_tile_str(validation_tiling_scheme)
        self.validation_name = "validation"
        self.validation_dp = os.path.join(self.workspace_dp, self.validation_name)
        self.validation_data_dp = os.path.join(self.validation_dp, "data")
        self.validation_images_dp = os.path.join(self.validation_dp, "images")
        self.validation_labels_dp = os.path.join(self.validation_dp, "labels")
        # The validation data resides in the test directory to reduce the
        # redundancy of datasets
        self.validation_datasets = self._get_datasets(
            self.validation_data_dp, requested_test_datasets
        )
        self.validation_dataset_path_manager = {
            validation_dataset.dn: TestDatasetPathManager(
                dataset=validation_dataset,
                tile_str=self.validation_tile_str,
                test_dp=self.validation_dp,
                test_data_dp=self.validation_data_dp,
                test_images_dp=self.validation_images_dp,
                test_labels_dp=self.validation_labels_dp,
                images_cover_csv_fn=self.images_cover_csv_fn,
            )
            for validation_dataset in self.validation_datasets
        }


        # =========================== Test Data ===========================
        self.test_tile_str = self._get_tile_str(test_tiling_scheme)

        self.test_name = "test"
        self.test_dp = os.path.join(self.workspace_dp, self.test_name)
        self.test_data_dp = os.path.join(self.test_dp, "data")
        self.test_images_dp = os.path.join(self.test_dp, "images")
        self.test_masks_dp = os.path.join(self.test_dp, "masks")
        self.test_labels_dp = os.path.join(self.test_dp, "labels")
        self.test_masks_aggregated_dp = os.path.join(
            self.test_dp, "masks_aggregated"
        )
        self.test_masks_labels_comparison_dp = os.path.join(
            self.test_dp, "masks_labels_comparison"
        )
        self.test_masks_labels_comparison_aggregated_dp = os.path.join(
            self.test_dp, "masks_labels_comparison_aggregated"
        )

        self.test_masks_fusion_comparison_dp = os.path.join(
            self.test_dp, "masks_fusion_comparison"
        )
        self.test_masks_fusion_comparison_aggregated_dp = os.path.join(
            self.test_dp, "masks_fusion_comparison_aggregated"
        )

        self.test_datasets = self._get_datasets(
            self.test_data_dp, requested_test_datasets
        )
        self.test_dataset_path_manager = {
            test_dataset.dn: TestDatasetPathManager(
                dataset=test_dataset,
                tile_str=self.test_tile_str,
                test_dp=self.test_dp,
                test_data_dp=self.test_data_dp,
                test_images_dp=self.test_images_dp,
                test_labels_dp=self.test_labels_dp,
                images_cover_csv_fn=self.images_cover_csv_fn,
                model_specific_dp=self.model_specific_dp,
                base_tile_prediction=prediction_base_tile_prediction,
                base_tile_merging=prediction_base_tile_merging,
                model_checkpoint_fn=prediction_model_checkpoint_fn,
                test_masks_dp=self.test_masks_dp,
                test_masks_aggregated_dp=self.test_masks_aggregated_dp,
                test_masks_labels_comparison_dp=self.test_masks_labels_comparison_dp,
                test_masks_labels_comparison_aggregated_dp=self.test_masks_labels_comparison_aggregated_dp,
                test_masks_fusion_comparison_dp=self.test_masks_fusion_comparison_dp,
                test_masks_fusion_comparison_aggregated_dp=self.test_masks_fusion_comparison_aggregated_dp,
            )
            for test_dataset in self.test_datasets
        }

    @classmethod
    def _get_tile_str(cls, tiling_scheme):
        tile_str = ""
        if tiling_scheme.represents_mercator_tiling():
            tile_str += "mercator"
            if tiling_scheme.get_zoom_level() is not None:
                tile_str += f"_zoom_level_{tiling_scheme.get_zoom_level()}"

        if tiling_scheme.represents_local_image_tiling():
            tile_str += "local"
            if tiling_scheme.is_centered_to_image():
                tile_str += "_centered"
            elif tiling_scheme.is_aligned_to_image_border():
                tile_str += "_border_aligned"
            elif tiling_scheme.is_optimal_aligned():
                tile_str += "_optimal_aligned"
            else:
                assert False
            if tiling_scheme.uses_overhanging_tiles():
                tile_str += f"_overhang_y"
            else:
                tile_str += f"_overhang_n"

        if tiling_scheme.uses_border_tiles():
            tile_str += f"_border_y"
        else:
            tile_str += f"_border_n"

        if tiling_scheme.represents_local_image_tiling():
            if tiling_scheme.is_in_pixel():
                tile_size_pixel = tiling_scheme.get_tile_size_in_pixel(True)
                tile_str += (
                    f"_size_p_{tile_size_pixel[0]}_{tile_size_pixel[1]}"
                )
                tile_stride_pixel = tiling_scheme.get_tile_stride_in_pixel(
                    True
                )
                tile_str += (
                    f"_stride_p_{tile_stride_pixel[0]}_{tile_stride_pixel[1]}"
                )
            elif tiling_scheme.is_in_meter():
                tile_size_meter = tiling_scheme.get_tile_size_in_meter(True)
                tile_str += (
                    f"_size_m_{tile_size_meter[0]}_{tile_size_meter[1]}"
                )
                tile_stride_meter = tiling_scheme.get_tile_stride_in_meter(
                    True
                )
                tile_str += (
                    f"_stride_m_{tile_stride_meter[0]}_{tile_stride_meter[1]}"
                )
            else:
                assert False

        return tile_str

    @staticmethod
    def _get_categories_str(categories, category_str_len=5):
        categories = [category[:category_str_len] for category in categories]
        categories_str = "-".join(categories)
        return categories_str

    @staticmethod
    def _get_datasets_str_list(datasets, join_datasets=True):
        if join_datasets:
            datasets = [dataset.dn for dataset in datasets]
            datasets_str_list = ["-".join(datasets)]
        else:
            datasets_str_list = [dataset.dn for dataset in datasets]
        return datasets_str_list

    @classmethod
    def get_model_specific_dp_list(
        cls,
        training_category_titles,
        train_datasets,
        train_tile_str,
        mmsegmentation_cfg_fp,
        mmsegmentation_cfg_description_str,
        join_datasets,
    ):
        training_category_str = cls._get_categories_str(
            training_category_titles
        )
        training_dataset_str_list = cls._get_datasets_str_list(
            train_datasets, join_datasets=join_datasets
        )
        model_specific_dp_list = []
        for training_dataset_str in training_dataset_str_list:
            model_str = os.path.splitext(
                os.path.basename(mmsegmentation_cfg_fp)
            )[0]
            model_specific_dp = os.path.join(
                training_category_str,
                training_dataset_str,
                train_tile_str,
                model_str,
                mmsegmentation_cfg_description_str,
            )
            model_specific_dp_list.append(model_specific_dp)
        return model_specific_dp_list

    @staticmethod
    def get_trained_model_dp(train_dp, model_specific_dp):
        return os.path.join(
            train_dp, "model", model_specific_dp
        )

    @staticmethod
    def get_prediction_model_checkpoint_fp(
        train_model_dp, prediction_model_checkpoint_fn
    ):
        prediction_model_checkpoint_fp = os.path.join(
            train_model_dp, prediction_model_checkpoint_fn
        )
        return prediction_model_checkpoint_fp

    def get_available_prediction_model_checkpoint_fp_list(
        self,
    ):
        model_specific_dp_list = self.get_model_specific_dp_list(
            self.training_category_titles,
            self.train_datasets,
            self.train_tile_str,
            self.segmentation_model_cfg_fp,
            self.mmsegmentation_cfg_description_str,
            join_datasets=False
        )
        prediction_model_checkpoint_fp_list = []
        for model_specific_dp in model_specific_dp_list:
            train_model_dp = self.get_trained_model_dp(
                self.train_dp, model_specific_dp
            )
            prediction_model_checkpoint_fp = self.get_prediction_model_checkpoint_fp(
                train_model_dp, self.prediction_model_checkpoint_fn
            )
            prediction_model_checkpoint_fp_list.append(
                prediction_model_checkpoint_fp
            )
        return prediction_model_checkpoint_fp_list


    @staticmethod
    def _get_dataset_paths(data_dp):
        return get_subdirs(data_dp, base_name_only=False, recursive=False)

    @classmethod
    def _get_datasets(cls, data_dp, requested_datasets):
        available_dataset_names = [
            os.path.basename(ifp) for ifp in cls._get_dataset_paths(data_dp)
        ]
        requested_dataset_names = [
            dataset.dn for dataset in requested_datasets
        ]
        requested_set = set(requested_dataset_names)
        available_set = set(available_dataset_names)
        if not requested_set.issubset(available_set):
            for requested_dataset in requested_set:
                err_msg = f"{requested_dataset} is not part of {available_set}"
                assert requested_dataset in available_set, err_msg
        return requested_datasets

    @property
    def active_test_datasets(self):
        if self.active_test_dataset_names is None:
            required_datasets = self.test_datasets
        else:
            required_datasets = [
                test_dataset
                for test_dataset in self.test_datasets
                if test_dataset.dn in self.active_test_dataset_names
            ]
        return required_datasets
