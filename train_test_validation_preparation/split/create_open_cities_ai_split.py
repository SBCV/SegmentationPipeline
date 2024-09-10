from train_test_validation_preparation.split.utility import split_dataset

if __name__ == "__main__":

    data_idp = "/mnt/Data-2TB/neat_eo_training/data_jointly/open_cities/data"

    # Option 1: All images
    data_split_odp = (
        "/mnt/Data-2TB/neat_eo_training/data_train_test/open_cities_split"
    )
    train_validation_test_ratio = (0.7, 0.1, 0.2)
    max_num_elements = None

    # Option 2: Tiny subset
    # data_split_odp = (
    #     "/mnt/Data-2TB/neat_eo_training/data_train_test/open_cities_split_tiny"
    # )
    # train_validation_test_ratio = (0.7, 0.1, 0.2)
    # max_num_elements = 5

    split_dataset(
        data_idp,
        data_split_odp,
        "open_cities",
        train_validation_test_ratio,
        max_num_elements,
    )
