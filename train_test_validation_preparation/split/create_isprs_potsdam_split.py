from train_test_validation_preparation.split.utility import split_dataset

if __name__ == "__main__":

    data_idp = "/mnt/DATA1-8TB/data_jointly/isprs_potsdam/data"

    # Option 1: All images
    data_split_odp = (
        "/mnt/DATA1-8TB/data_train_validation_test/isprs_potsdam_train_validation_test"
    )
    train_validation_test_ratio = (0.7, 0.1, 0.2)
    max_num_elements = None

    # Option 2: Tiny subset
    # data_split_odp = (
    #     "/mnt/Data-2TB/neat_eo_training/data_train_test/isprs_potsdam_split_tiny"
    # )
    # train_test_ratio = 0.8
    # max_num_elements = 5

    dataset_name = "potsdam"
    dataset_name += f"_{train_validation_test_ratio[0]}"
    dataset_name += f"_{train_validation_test_ratio[1]}"
    dataset_name += f"_{train_validation_test_ratio[2]}"

    split_dataset(
        data_idp,
        data_split_odp,
        dataset_name,
        train_validation_test_ratio,
        max_num_elements,
    )
