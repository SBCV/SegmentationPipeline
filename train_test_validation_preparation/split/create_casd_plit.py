from train_test_validation_preparation.split.utility import split_dataset

if __name__ == "__main__":

    data_idp = "/mnt/Data-2TB/neat_eo_training/data_jointly/casd/data"

    data_split_odp = (
        "/mnt/Data-2TB/neat_eo_training/data_train_test/casd_split"
    )
    train_validation_test_ratio = (0.7, 0.1, 0.2)
    max_num_elements = None

    split_dataset(
        data_idp,
        data_split_odp,
        "casd",
        train_validation_test_ratio,
        max_num_elements,
    )
