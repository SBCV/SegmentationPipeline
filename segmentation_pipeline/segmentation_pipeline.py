from segmentation_pipeline.data_pipeline import DataPipeline


class SegmentationPipeline(DataPipeline):
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
        create_image_json_vis=False,
        create_label_json_vis=False,
        create_tile_aux_files=False,
        debug_prediction_merging=False,
        debug_prediction_create_mmseg_pkl_file=False,
        debug_prediction_load_mmseg_pkl_file=False,
        debug_prediction_max_number_tiles_per_image=None,
        clear_split_data=False,
        aggregate_as_images=False,
        aggregate_as_images_with_pixel_projection=True,
        aggregate_as_json=True,
        aggregate_as_global_json=False,
        aggregate_save_normalized_raster=False,
        eval_metrics=None,
        resume_training=False,
        deterministic_training=False,
        disable_train_pipeline_vertical_flip=False,
        disable_train_pipeline_rotation=False,
        disable_train_pipeline_resizing=False,
        disable_test_pipeline_resizing=False,
        ignore_background_category_in_evaluation=False,
        perform_prediction_base_tile_prediction=False,
        perform_prediction_base_tile_merging=False,
        invalid_category_target_index=0,
        lazy=False,
    ):
        super().__init__(
            pm=pm,
            image_band_entries=image_band_entries,
            label_band_entries=label_band_entries,
            multi_dataset_categories=multi_dataset_categories,
            no_data_threshold=no_data_threshold,
            output_tile_size_pixel=output_tile_size_pixel,
            train_tiling_scheme=train_tiling_scheme,
            test_tiling_scheme=test_tiling_scheme,
            validation_tiling_scheme=validation_tiling_scheme,
            create_image_json_vis=create_image_json_vis,
            create_label_json_vis=create_label_json_vis,
            create_tile_aux_files=create_tile_aux_files,
            debug_prediction_max_number_tiles_per_image=debug_prediction_max_number_tiles_per_image,
            clear_split_data=clear_split_data,
            aggregate_as_images=aggregate_as_images,
            aggregate_as_images_with_pixel_projection=aggregate_as_images_with_pixel_projection,
            aggregate_as_json=aggregate_as_json,
            aggregate_as_global_json=aggregate_as_global_json,
            aggregate_save_normalized_raster=aggregate_save_normalized_raster,
            lazy=lazy
        )

        self.invalid_category_target_index = invalid_category_target_index

        if self.train_tiling_scheme.represents_local_image_tiling():
            self.train_tiling_scheme.set_aligned_to_base_tile_area_flag(False)

        if self.test_tiling_scheme.represents_local_image_tiling():
            self.test_tiling_scheme.set_aligned_to_base_tile_area_flag(True)

        if self.validation_tiling_scheme.represents_local_image_tiling():
            self.validation_tiling_scheme.set_aligned_to_base_tile_area_flag(
                True
            )

        self.debug_prediction_merging = debug_prediction_merging
        self.debug_prediction_create_mmseg_pkl_file = (
            debug_prediction_create_mmseg_pkl_file
        )
        self.debug_prediction_load_mmseg_pkl_file = (
            debug_prediction_load_mmseg_pkl_file
        )
        self.eval_metrics = eval_metrics

        self.resume_training = resume_training
        self.deterministic_training = deterministic_training
        self.disable_train_pipeline_vertical_flip = (
            disable_train_pipeline_vertical_flip
        )
        self.disable_train_pipeline_rotation = disable_train_pipeline_rotation
        self.disable_train_pipeline_resizing = disable_train_pipeline_resizing
        self.disable_test_pipeline_resizing = disable_test_pipeline_resizing

        self.ignore_background_category_in_evaluation = (
            ignore_background_category_in_evaluation
        )
        self.perform_prediction_base_tile_prediction = (
            perform_prediction_base_tile_prediction
        )
        self.perform_prediction_base_tile_merging = (
            perform_prediction_base_tile_merging
        )

    def train(self):
        raise NotImplementedError

    def compute_predictions(self, checkpoint_fp=None):
        raise NotImplementedError

    def evaluate_predictions(self):
        raise NotImplementedError
