import os
import shutil

from eot.tools.tools_api import (
    run_tile_images,
    run_cover,
    run_aggregate,
    run_rasterize,
)
from eot.aggregation.tile_aggregation import (
    aggregate_dataset_tile_predictions_per_raster,
)
from eot.geojson_ext.geojson_creation import (
    create_geojson_for_image_tiles,
    create_geojson_for_label_tiles,
)
from eot.categories.dataset_category import DatasetCategory
from eot.categories.dataset_categories import DatasetCategories
from eot.comparison.category_comparison import CategoryComparison
from eot.tools.tools_api import run_compare
from eot.utility.os_ext import makedirs_safely

from neat_eo.utility.download_utility import extract_file
from neat_eo.utility.os_extension import mkdir_safely


class DataPipeline:

    def __init__(
        self,
        pm,
        image_band_entries,
        label_band_entries,
        multi_dataset_categories,
        no_data_threshold,
        output_tile_size_pixel,
        train_tiling_scheme,
        test_tiling_scheme,
        validation_tiling_scheme,
        create_image_json_vis,
        create_label_json_vis,
        create_tile_aux_files,
        debug_prediction_max_number_tiles_per_image,
        clear_split_data,
        aggregate_as_images,
        aggregate_as_images_with_pixel_projection,
        aggregate_as_json,
        aggregate_as_global_json,
        aggregate_save_normalized_raster,
        lazy=False,
    ):
        self.pm = pm

        self.multi_dataset_categories = multi_dataset_categories
        self.train_tiling_scheme = train_tiling_scheme
        self.test_tiling_scheme = test_tiling_scheme
        self.validation_tiling_scheme = validation_tiling_scheme

        self.output_tile_size_pixel = output_tile_size_pixel
        self.dataset_to_image_bands = {
            entry.dn: entry.bands for entry in image_band_entries
        }
        self.dataset_to_label_bands = {
            entry.dn: entry.bands for entry in label_band_entries
        }
        self.no_data_threshold = no_data_threshold
        self.create_image_json_vis = create_image_json_vis
        self.create_label_json_vis = create_label_json_vis
        self.create_tile_aux_files = create_tile_aux_files

        self.debug_prediction_max_number_tiles_per_image = (
            debug_prediction_max_number_tiles_per_image
        )
        self.clear_split_data = clear_split_data
        self.aggregate_as_images = aggregate_as_images
        self.aggregate_as_images_with_pixel_projection = (
            aggregate_as_images_with_pixel_projection
        )
        self.aggregate_as_json = aggregate_as_json
        self.aggregate_as_global_json = aggregate_as_global_json
        self.aggregate_save_normalized_raster = (
            aggregate_save_normalized_raster
        )

        # fmt:off
        self.comparison_categories = DatasetCategories(
            [
                DatasetCategory(name=CategoryComparison.true_positive_name, palette_index=1, palette_color=(0, 0, 127)),        # noqa
                DatasetCategory(name=CategoryComparison.false_positive_name, palette_index=2, palette_color=(0, 127, 0)),       # noqa
                DatasetCategory(name=CategoryComparison.false_negative_name, palette_index=4, palette_color=(127, 0, 0)),       # noqa
                DatasetCategory(name=CategoryComparison.true_negative_name, palette_index=3, palette_color=(127, 127, 127)),    # noqa
            ]
        )
        # fmt:on

        self.lazy = lazy

    def retrieve_dataset_from_folder(self, dataset_dp):
        dataset_dp = os.path.abspath(dataset_dp)
        dataset_train_dp = os.path.join(dataset_dp, self.pm.train_name)
        dataset_predict_dp = os.path.join(dataset_dp, self.pm.test_name)
        assert os.path.isdir(dataset_train_dp)
        assert os.path.isdir(dataset_predict_dp)
        assert not os.path.isdir(self.pm.train_dp)
        assert not os.path.isdir(self.pm.test_dp)
        shutil.copytree(dataset_train_dp, self.pm.train_dp)
        shutil.copytree(dataset_predict_dp, self.pm.test_dp)

    def retrieve_dataset_from_archive(self, tar_fp):
        extract_file(
            tar_fp,
            self.pm.workspace_dp,
            lazy=self.lazy,
            lazy_check_paths=[self.pm.train_dp, self.pm.test_dp],
        )

    def _tile_images(
        self,
        tif_dp,
        tile_odp,
        dataset_type,
        tiling_scheme,
        debug_max_number_tiles_per_image=None,
    ):
        # ==================
        # Tile Imagery:
        # ==================
        search_regex = self.pm.dataset_to_image_search_regex[dataset_type]
        ignore_regex = self.pm.dataset_to_image_ignore_regex[dataset_type]
        run_tile_images(
            tif_idp=tif_dp,
            tif_search_regex=search_regex,
            tif_ignore_regex=ignore_regex,
            tile_odp=tile_odp,
            bands=self.dataset_to_image_bands[dataset_type],
            output_tile_size_pixel=self.output_tile_size_pixel,
            tiling_scheme=tiling_scheme,
            no_data_threshold=self.no_data_threshold,
            create_aux_files=self.create_tile_aux_files,
            clear_split_data=self.clear_split_data,
            debug_max_number_tiles_per_image=debug_max_number_tiles_per_image,
            lazy=self.lazy,
        )

    def _compute_label_cover(self, images_dp, cover_csv_fp):
        # Retrieve and tile labels accordingly:
        run_cover(tile_idp=images_dp, ofp_list=[cover_csv_fp], lazy=self.lazy)

    def _tile_labels(
        self,
        raster_or_geojson_idp,
        cover_csv_ifp,
        label_odp,
        dataset_type,
        tiling_scheme,
        debug_max_number_tiles_per_image=None,
    ):
        search_regex = self.pm.dataset_to_label_search_regex[dataset_type]
        ignore_regex = self.pm.dataset_to_label_ignore_regex[dataset_type]
        label_ext = os.path.splitext(
            self.pm.dataset_to_label_search_regex[dataset_type]
        )[1]
        categories = self.multi_dataset_categories.get_dataset_categories(
            dataset_type
        )
        if label_ext == ".geojson":
            run_rasterize(
                geojson_idp=raster_or_geojson_idp,
                geojson_search_regex=search_regex,
                geojson_ignore_regex=ignore_regex,
                cover_csv_ifp=cover_csv_ifp,
                categories=categories,
                label_odp=label_odp,
                lazy=self.lazy,
            )
        else:
            run_tile_images(
                tif_idp=raster_or_geojson_idp,
                tif_search_regex=search_regex,
                tif_ignore_regex=ignore_regex,
                tile_odp=label_odp,
                bands=self.dataset_to_label_bands[dataset_type],
                output_tile_size_pixel=self.output_tile_size_pixel,
                tiling_scheme=tiling_scheme,
                write_labels=True,
                cover_csv_ifp=cover_csv_ifp,
                convert_images_to_labels=True,
                categories=categories,
                # Don't apply the "no_data_threshold" to labels
                no_data_threshold=100,
                create_aux_files=self.create_tile_aux_files,
                clear_split_data=self.clear_split_data,
                debug_max_number_tiles_per_image=debug_max_number_tiles_per_image,
                lazy=self.lazy,
            )

        # TODO
        # split_panoptic_json()

    def _prepare_dataset_training_data(self, tdpm):
        # Here: tdpm = train dataset path manager
        assert os.path.isdir(tdpm.data_dp), f"{tdpm.data_dp}"
        tiling_scheme = self.train_tiling_scheme
        self._tile_images(
            tdpm.data_dp,
            tdpm.images_dp,
            tdpm.dataset_type,
            tiling_scheme=tiling_scheme,
        )
        self._compute_label_cover(
            tdpm.images_dp, tdpm.images_cover_csv_fp
        )
        self._tile_labels(
            raster_or_geojson_idp=tdpm.data_dp,
            cover_csv_ifp=tdpm.images_cover_csv_fp,
            label_odp=tdpm.labels_dp,
            dataset_type=tdpm.dataset_type,
            tiling_scheme=tiling_scheme,
        )

        # if self.create_image_json_vis:
        #     self._create_image_json_vis(
        #         tdpm.train_images_dp, tdpm.train_data_dp
        #     )

    def prepare_training_data(self):
        for train_dataset in self.pm.train_datasets:
            train_dpm = self.pm.train_dataset_path_manager[train_dataset.dn]
            self._prepare_dataset_training_data(train_dpm)

        for validation_dataset in self.pm.validation_datasets:
            validation_dpm = self.pm.validation_dataset_path_manager[
                validation_dataset.dn
            ]
            self._prepare_dataset_test_or_validation_data(
                validation_dpm,
                self.validation_tiling_scheme,
            )

    def _prepare_dataset_test_or_validation_data(
        self,
        tdpm,
        tiling_scheme,
        skip_label_data=False,
        debug_max_number_tiles_per_image=None,
    ):
        search_regex = self.pm.dataset_to_image_search_regex[tdpm.dataset_type]
        ignore_regex = self.pm.dataset_to_image_ignore_regex[tdpm.dataset_type]
        categories = self.multi_dataset_categories.get_dataset_categories(
            tdpm.dataset_type
        )
        # Here: tdpm = test dataset path manager
        assert os.path.isdir(tdpm.data_dp), f"{tdpm.data_dp}"
        self._tile_images(
            tdpm.data_dp,
            tdpm.images_dp,
            tdpm.dataset_type,
            tiling_scheme=tiling_scheme,
            debug_max_number_tiles_per_image=debug_max_number_tiles_per_image,
        )
        if self.create_image_json_vis:
            create_geojson_for_image_tiles(
                data_dp=tdpm.data_dp,
                search_regex=search_regex,
                ignore_regex=ignore_regex,
                images_dp=tdpm.images_dp,
                grid_json_fn=self.pm.grid_json_fn,
            )
        self._compute_label_cover(
            tdpm.images_dp, tdpm.images_cover_csv_fp
        )
        if skip_label_data:
            return
        self._tile_labels(
            raster_or_geojson_idp=tdpm.data_dp,
            cover_csv_ifp=tdpm.images_cover_csv_fp,
            label_odp=tdpm.labels_dp,
            dataset_type=tdpm.dataset_type,
            tiling_scheme=tiling_scheme,
            debug_max_number_tiles_per_image=debug_max_number_tiles_per_image,
        )
        if self.create_label_json_vis:
            create_geojson_for_label_tiles(
                test_data_dp=tdpm.data_dp,
                search_regex=search_regex,
                ignore_regex=ignore_regex,
                test_images_dp=tdpm.images_dp,
                test_labels_dp=tdpm.labels_dp,
                categories=categories,
                odp=tdpm.labels_dp,
                grid_json_fn=self.pm.grid_json_fn,
                lazy=self.lazy,
            )

    def prepare_test_data(self, skip_label_data=False):
        for test_dataset in self.pm.active_test_datasets:
            test_dpm = self.pm.test_dataset_path_manager[test_dataset.dn]
            self._prepare_dataset_test_or_validation_data(
                test_dpm,
                self.test_tiling_scheme,
                skip_label_data=skip_label_data,
                debug_max_number_tiles_per_image=self.debug_prediction_max_number_tiles_per_image,
            )

    def _create_aggregation_odp(self):
        mkdir_safely(self.pm.masks_aggregated_dp)

    def _aggregate_tile_predictions_per_raster(self):
        self._create_aggregation_odp()
        for test_dataset in self.pm.active_test_datasets:
            tdpm = self.pm.test_dataset_path_manager[test_dataset.dn]
            search_regex = self.pm.dataset_to_image_search_regex[
                tdpm.dataset_type
            ]
            ignore_regex = self.pm.dataset_to_image_ignore_regex[
                tdpm.dataset_type
            ]
            categories = self.multi_dataset_categories.get_dataset_categories(
                tdpm.dataset_type
            )
            aggregate_dataset_tile_predictions_per_raster(
                aggregation_odp=tdpm.masks_aggregated_dp,
                test_data_dp=tdpm.data_dp,
                test_masks_dp=tdpm.masks_dp,
                search_regex=search_regex,
                ignore_regex=ignore_regex,
                mercator_tiling_flag=self.test_tiling_scheme.represents_mercator_tiling(),
                test_data_normalized_dp=tdpm.data_normalized_dp,
                aggregate_save_normalized_raster=self.aggregate_save_normalized_raster,
                aggregate_as_json=self.aggregate_as_json,
                aggregate_as_images=self.aggregate_as_images,
                use_pixel_projection=self.aggregate_as_images_with_pixel_projection,
                categories=categories.get_non_ignore_categories(),
                grid_json_fn=self.pm.grid_json_fn,
                lazy=self.lazy,
            )

    def _aggregate_global_mercator_tile_predictions(self):
        # Create for each dataset a single (global) json file reflecting the
        # information the corresponding geo-tiles (web map)
        for test_dataset in self.pm.test_datasets:
            tdpm = self.pm.test_dataset_path_manager[test_dataset.dn]
            categories = self.multi_dataset_categories.get_dataset_categories(
                tdpm.dataset_type
            )
            run_aggregate(
                masks_idp=tdpm.masks_dp,
                categories=categories.get_non_ignore_categories(),
                geojson_odp=self.pm.masks_aggregated_dp,
                geojson_grid_ofn=self.pm.grid_json_fn,
                lazy=self.lazy,
            )

    def aggregate_predictions(self):
        if self.test_tiling_scheme.represents_mercator_tiling():
            if self.aggregate_as_global_json:
                self._aggregate_global_mercator_tile_predictions()
        self._aggregate_tile_predictions_per_raster()

    def compare_test_predictions_with_labels(self):
        # =====================================================================
        # Compare the trained model prediction against the corresponding labels
        # =====================================================================
        for test_dataset in self.pm.test_datasets:
            tdpm = self.pm.test_dataset_path_manager[test_dataset.dn]
            comparison_odp = tdpm.masks_labels_comparison_dp
            comparison_aggregated_odp = (
                tdpm.masks_labels_comparison_aggregated_dp
            )
            segmentation_categories = (
                self.multi_dataset_categories.get_dataset_categories(
                    tdpm.dataset_type
                )
            )

            run_compare(
                comparison_odp=comparison_odp,
                segmentation_categories=segmentation_categories,
                comparison_categories=self.comparison_categories,
                label_idp=tdpm.labels_dp,
                mask_idp=tdpm.masks_dp,
                lazy=self.lazy,
            )

            search_regex = self.pm.dataset_to_image_search_regex[
                tdpm.dataset_type
            ]
            ignore_regex = self.pm.dataset_to_image_ignore_regex[
                tdpm.dataset_type
            ]
            categories = self.multi_dataset_categories.get_dataset_categories(
                tdpm.dataset_type
            )
            makedirs_safely(comparison_aggregated_odp)
            for (
                segmentation_category
            ) in (
                categories.get_active_categories().get_non_ignore_categories()
            ):
                category_tile_comparison_dp = os.path.join(
                    comparison_odp, segmentation_category.name
                )
                category_tile_comparison_aggregated_odp = os.path.join(
                    comparison_aggregated_odp, segmentation_category.name
                )
                makedirs_safely(category_tile_comparison_aggregated_odp)

                aggregate_dataset_tile_predictions_per_raster(
                    aggregation_odp=category_tile_comparison_aggregated_odp,
                    test_data_dp=tdpm.data_dp,
                    test_masks_dp=category_tile_comparison_dp,
                    search_regex=search_regex,
                    ignore_regex=ignore_regex,
                    mercator_tiling_flag=self.test_tiling_scheme.represents_mercator_tiling(),
                    test_data_normalized_dp=tdpm.data_normalized_dp,
                    aggregate_save_normalized_raster=self.aggregate_save_normalized_raster,
                    aggregate_as_json=self.aggregate_as_json,
                    aggregate_as_images=self.aggregate_as_images,
                    use_pixel_projection=self.aggregate_as_images_with_pixel_projection,
                    categories=self.comparison_categories,
                    grid_json_fn=self.pm.grid_json_fn,
                    lazy=False,
                )

    def compare_fusion_with_base_predictions(self):
        # =====================================================================
        # Compare the fused predictions with the standard base predictions
        # =====================================================================
        for test_dataset in self.pm.test_datasets:
            tdpm = self.pm.test_dataset_path_manager[test_dataset.dn]
            comparison_odp = tdpm.masks_fusion_comparison_dp
            comparison_aggregated_odp = (
                tdpm.masks_fusion_comparison_aggregated_dp
            )
            segmentation_categories = (
                self.multi_dataset_categories.get_dataset_categories(
                    tdpm.dataset_type
                )
            )

            run_compare(
                comparison_odp=comparison_odp,
                segmentation_categories=segmentation_categories,
                comparison_categories=self.comparison_categories,
                label_idp=tdpm.masks_base_dp,
                mask_idp=tdpm.masks_dp,
                lazy=self.lazy,
            )

            search_regex = self.pm.dataset_to_image_search_regex[
                tdpm.dataset_type
            ]
            ignore_regex = self.pm.dataset_to_image_ignore_regex[
                tdpm.dataset_type
            ]
            categories = self.multi_dataset_categories.get_dataset_categories(
                tdpm.dataset_type
            )
            makedirs_safely(comparison_aggregated_odp)
            for (
                segmentation_category
            ) in (
                categories.get_active_categories().get_non_ignore_categories()
            ):
                category_tile_comparison_dp = os.path.join(
                    comparison_odp, segmentation_category.name
                )
                category_tile_comparison_aggregated_odp = os.path.join(
                    comparison_aggregated_odp, segmentation_category.name
                )
                makedirs_safely(category_tile_comparison_aggregated_odp)

                aggregate_dataset_tile_predictions_per_raster(
                    aggregation_odp=category_tile_comparison_aggregated_odp,
                    test_data_dp=tdpm.data_dp,
                    test_masks_dp=category_tile_comparison_dp,
                    search_regex=search_regex,
                    ignore_regex=ignore_regex,
                    mercator_tiling_flag=self.test_tiling_scheme.represents_mercator_tiling(),
                    test_data_normalized_dp=tdpm.data_normalized_dp,
                    aggregate_save_normalized_raster=self.aggregate_save_normalized_raster,
                    aggregate_as_json=self.aggregate_as_json,
                    aggregate_as_images=self.aggregate_as_images,
                    use_pixel_projection=self.aggregate_as_images_with_pixel_projection,
                    categories=self.comparison_categories,
                    grid_json_fn=self.pm.grid_json_fn,
                    lazy=False,
                )
