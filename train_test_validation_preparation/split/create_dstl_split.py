from train_test_validation_preparation.split.utility import (
    split_dataset,
    create_splitting_report,
)
from neat_eo_data.preparation.datasets.dstl.dstl_class_info import (
    CLASSES,
    COLORS,
)


def _compute_filter_label_values(filter_category_names):
    if filter_category_names is None:
        class_name_to_class_index = None
    else:
        class_name_to_class_index = {v: k for k, v in CLASSES.items()}
        filter_labels_values = [
            COLORS[class_name_to_class_index[name]]
            for name in filter_category_names
        ]
    return class_name_to_class_index


def main():
    data_idp = "/mnt/Data-2TB/neat_eo_training/data_jointly/dstl/data_filtered"

    # Option 1: All images
    data_split_odp = (
        "/mnt/Data-2TB/neat_eo_training/data_train_test/dstl_filtered_split"
    )
    train_validation_test_ratio = (0.7, 0.1, 0.2)
    max_num_elements = None

    # Option 2: Tiny subset
    # data_split_odp = (
    #     "/mnt/Data-2TB/neat_eo_training/data_train_test/dstl_filtered_split_tiny"
    # )
    # train_validation_test_ratio = (0.7, 0.1, 0.2)
    # max_num_elements = 5

    # filter_category_names = ["Building", "Road", "Trees"]
    filter_category_names = None
    aggregator_func = all

    data_split_odp = data_split_odp + f"_{train_validation_test_ratio}"
    filter_labels_values = _compute_filter_label_values(filter_category_names)

    def mask_entry_func(entry):
        if filter_labels_values is not None:
            aggregated_result = entry.contains_label_values(
                filter_labels_values, aggregator_func
            )
            if not aggregated_result:
                entry.masked = True

    split_dataset(
        data_idp,
        data_split_odp,
        "dstl",
        train_validation_test_ratio,
        max_num_elements,
        mask_entry_func=mask_entry_func,
    )

    create_splitting_report(
        data_split_odp,
        train_test_ratio=train_validation_test_ratio,
        max_num_elements=max_num_elements,
        filter_category_names=filter_category_names,
        aggregator_func=aggregator_func,
    )


if __name__ == "__main__":
    main()
