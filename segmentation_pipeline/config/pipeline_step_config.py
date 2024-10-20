from typing import Union
from pydantic import BaseModel


class PipelineStepConfig(BaseModel):
    process_semantic_dataset: Union[bool, int] = None

    prepare_semantic_training_data: Union[bool, int]
    prepare_semantic_test_data: Union[bool, int]

    # Panoptic Processing Option 1
    derive_panoptic_train_labels_from_raster: Union[bool, int] = None
    derive_panoptic_test_labels_from_raster: Union[bool, int] = None

    # Panoptic Processing Option 2
    prepare_panoptic_training_data: Union[bool, int]
    prepare_panoptic_test_data: Union[bool, int]
    derive_panoptic_train_labels_from_tiles: Union[bool, int] = None
    derive_panoptic_test_labels_from_tiles: Union[bool, int] = None

    run_training: Union[bool, int]
    export_model_complexity: Union[bool, int] = None

    compute_predictions: Union[bool, int]
    show_results_interactive: Union[bool, int] = None
    evaluate_predictions: Union[bool, int]
    aggregate_predictions: Union[bool, int]
    compare_predictions_with_labels: Union[bool, int] = None
    compare_fusion_with_base_predictions: Union[bool, int] = None

    lazy: Union[bool, int]

    resume_training: Union[bool, int] = False
    deterministic_training: Union[bool, int] = False

    disable_train_pipeline_vertical_flip: Union[bool, int] = False
    disable_train_pipeline_rotation: Union[bool, int] = False
    disable_train_pipeline_resizing: Union[bool, int] = False
    disable_test_pipeline_resizing: Union[bool, int] = False

    perform_prediction_base_tile_prediction: Union[bool, int] = False
    perform_prediction_base_tile_merging: Union[bool, int] = False
    ignore_background_category_in_evaluation: Union[bool, int] = False

    eval_metrics: list

    aggregate_as_images: Union[bool, int] = True
    aggregate_as_images_with_pixel_projection: Union[bool, int] = True
    aggregate_as_json: Union[bool, int] = True
    aggregate_as_global_json: Union[bool, int] = True
    aggregate_save_normalized_raster: Union[bool, int] = True

    create_image_json_vis: Union[bool, int] = False
    create_label_json_vis: Union[bool, int] = False
    create_tile_aux_files: Union[bool, int] = False
    debug_prediction_merging: Union[bool, int] = False
    debug_prediction_create_mmseg_pkl_file: Union[bool, int] = False
    debug_prediction_load_mmseg_pkl_file: Union[bool, int] = False
    debug_prediction_max_number_tiles_per_image: int = None
    clear_split_data: Union[bool, int] = True
