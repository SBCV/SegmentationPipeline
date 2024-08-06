import os


class TestDatasetPathManager:
    def __init__(
        self,
        dataset,
        tile_str,
        test_dp,
        test_data_dp,
        test_images_dp,
        test_labels_dp,
        images_cover_csv_fn,
        model_specific_dp=None,
        base_tile_prediction=None,
        base_tile_merging=None,
        model_checkpoint_fn=None,
        test_masks_dp=None,
        test_masks_aggregated_dp=None,
        test_masks_labels_comparison_dp=None,
        test_masks_labels_comparison_aggregated_dp=None,
        test_masks_fusion_comparison_dp=None,
        test_masks_fusion_comparison_aggregated_dp=None,
    ):
        if test_masks_dp is None:
            test_masks_dp = str(test_masks_dp)
        if test_masks_aggregated_dp is None:
            test_masks_aggregated_dp = str(test_masks_aggregated_dp)
        if test_masks_labels_comparison_dp is None:
            test_masks_labels_comparison_dp = str(
                test_masks_labels_comparison_dp
            )
        if test_masks_fusion_comparison_dp is None:
            test_masks_fusion_comparison_dp = str(
                test_masks_fusion_comparison_dp
            )
        if test_masks_fusion_comparison_aggregated_dp is None:
            test_masks_fusion_comparison_aggregated_dp = str(
                test_masks_fusion_comparison_aggregated_dp
            )

        if test_masks_labels_comparison_aggregated_dp is None:
            test_masks_labels_comparison_aggregated_dp = str(
                test_masks_labels_comparison_aggregated_dp
            )

        self.dataset_name = dataset.dn
        self.dataset_type = dataset.dataset_type
        self.test_data_dp = os.path.join(test_data_dp, self.dataset_name)
        self.test_data_normalized_dp = os.path.join(
            test_dp, "data_normalized", self.dataset_name
        )
        self.test_images_dp = os.path.join(
            test_images_dp, self.dataset_name, tile_str
        )
        self.test_images_cover_csv_fp = os.path.join(
            self.test_images_dp, images_cover_csv_fn
        )
        self.test_labels_dp = os.path.join(
            test_labels_dp, self.dataset_name, tile_str
        )

        # The following paths refer to the results produced with a specific
        #  (trained) model.
        self.model_specific_dp = model_specific_dp
        if base_tile_prediction is not None:
            base_tile_prediction_str = (
                f"base_tile_prediction_{base_tile_prediction}"
            )
        else:
            base_tile_prediction_str = ""
        if base_tile_merging is not None:
            base_tile_merging_str = f"base_tile_merging_{base_tile_merging}"
        else:
            base_tile_merging_str = ""
        self.model_checkpoint_fn = model_checkpoint_fn
        if model_checkpoint_fn is not None:
            model_stem, model_ext = os.path.splitext(model_checkpoint_fn)
            assert model_ext == ".pth"
            model_checkpoint_str = model_stem
        else:
            model_checkpoint_str = ""
        model_specific_dp_str = str(self.model_specific_dp)

        test_relative_dp = os.path.join(
            self.dataset_name,
            tile_str,
            base_tile_prediction_str,
            base_tile_merging_str,
            model_specific_dp_str,
            model_checkpoint_str,
        )
        self.test_masks_dp = os.path.join(test_masks_dp, test_relative_dp)
        self.test_masks_cover_csv_fp = os.path.join(
            self.test_masks_dp, "masks_cover.csv"
        )
        self.test_masks_eval_dp = os.path.join(
            test_masks_dp + "_eval", test_relative_dp
        )
        self.test_masks_aggregated_dp = os.path.join(
            test_masks_aggregated_dp, test_relative_dp
        )

        # Paths required for comparison of predictions and labels
        self.test_masks_labels_comparison_dp = os.path.join(
            test_masks_labels_comparison_dp, test_relative_dp
        )
        self.test_masks_labels_comparison_aggregated_dp = os.path.join(
            test_masks_labels_comparison_aggregated_dp, test_relative_dp
        )

        # Paths required for comparison of fused and base predictions
        test_relative_base_dp = os.path.join(
            self.dataset_name,
            tile_str,
            base_tile_prediction_str,
            f"base_tile_merging_{False}",
            model_specific_dp_str,
            model_checkpoint_str,
        )
        self.test_masks_base_dp = os.path.join(
            test_masks_dp, test_relative_base_dp
        )

        self.test_masks_fusion_comparison_dp = os.path.join(
            test_masks_fusion_comparison_dp, test_relative_dp
        )
        self.test_masks_fusion_comparison_aggregated_dp = os.path.join(
            test_masks_fusion_comparison_aggregated_dp, test_relative_dp
        )
