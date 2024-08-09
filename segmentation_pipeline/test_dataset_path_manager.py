import os

from segmentation_pipeline.base_dataset_path_manager import BaseDatasetPathManager


class TestDatasetPathManager(BaseDatasetPathManager):
    def __init__(
        self,
        dataset,
        tile_str,
        root_dp,
        data_dp,
        images_dp,
        labels_dp,
        images_cover_csv_fn,
        model_specific_dp=None,
        base_tile_prediction=None,
        base_tile_merging=None,
        model_checkpoint_fn=None,
        masks_dp=None,
        masks_aggregated_dp=None,
        masks_labels_comparison_dp=None,
        masks_labels_comparison_aggregated_dp=None,
        masks_fusion_comparison_dp=None,
        masks_fusion_comparison_aggregated_dp=None,
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

        if masks_dp is None:
            masks_dp = str(masks_dp)
        if masks_aggregated_dp is None:
            masks_aggregated_dp = str(masks_aggregated_dp)
        if masks_labels_comparison_dp is None:
            masks_labels_comparison_dp = str(
                masks_labels_comparison_dp
            )
        if masks_fusion_comparison_dp is None:
            masks_fusion_comparison_dp = str(
                masks_fusion_comparison_dp
            )
        if masks_fusion_comparison_aggregated_dp is None:
            masks_fusion_comparison_aggregated_dp = str(
                masks_fusion_comparison_aggregated_dp
            )

        if masks_labels_comparison_aggregated_dp is None:
            masks_labels_comparison_aggregated_dp = str(
                masks_labels_comparison_aggregated_dp
            )

        self.data_normalized_dp = os.path.join(
            root_dp, "data_normalized", self.dataset_name
        )
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

        relative_dp = os.path.join(
            self.dataset_name,
            tile_str,
            base_tile_prediction_str,
            base_tile_merging_str,
            model_specific_dp_str,
            model_checkpoint_str,
        )
        self.masks_dp = os.path.join(masks_dp, relative_dp)
        self.masks_cover_csv_fp = os.path.join(
            self.masks_dp, "masks_cover.csv"
        )
        self.masks_eval_dp = os.path.join(
            masks_dp + "_eval", relative_dp
        )
        self.masks_aggregated_dp = os.path.join(
            masks_aggregated_dp, relative_dp
        )

        # Paths required for comparison of predictions and labels
        self.masks_labels_comparison_dp = os.path.join(
            masks_labels_comparison_dp, relative_dp
        )
        self.masks_labels_comparison_aggregated_dp = os.path.join(
            masks_labels_comparison_aggregated_dp, relative_dp
        )

        # Paths required for comparison of fused and base predictions
        relative_base_dp = os.path.join(
            self.dataset_name,
            tile_str,
            base_tile_prediction_str,
            f"base_tile_merging_{False}",
            model_specific_dp_str,
            model_checkpoint_str,
        )
        self.masks_base_dp = os.path.join(
            masks_dp, relative_base_dp
        )

        self.masks_fusion_comparison_dp = os.path.join(
            masks_fusion_comparison_dp, relative_dp
        )
        self.masks_fusion_comparison_aggregated_dp = os.path.join(
            masks_fusion_comparison_aggregated_dp, relative_dp
        )
