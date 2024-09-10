import os
from shutil import copytree

from train_test_validation_preparation.split.dataset import Dataset


def _copy_data_split(entries, odp):
    os.makedirs(odp)
    for entry in entries:
        print("entry", entry)
        image_idp = entry.image_dp
        label_idp = entry.label_dp
        image_odp = os.path.join(odp, os.path.basename(image_idp))
        label_odp = os.path.join(odp, os.path.basename(label_idp))
        copytree(image_idp, image_odp)
        copytree(label_idp, label_odp)


def split_dataset(
    idp,
    odp,
    dataset_name,
    train_validation_test_ratio,
    max_num_elements=None,
    mask_entry_func=None,
):
    assert idp != odp
    dataset = Dataset.retrieve_from_dp(idp)

    use_masks = mask_entry_func is not None
    if use_masks:
        dataset.mask_entries(mask_entry_func)

    for entry in dataset.get_entries():
        print(
            f"Ifp: {entry.image_fp} Masked: {entry.masked} label_values: {entry.get_label_values()}"
        )

    (
        train_entries,
        validation_entries,
        test_entries,
    ) = dataset.split_data_by_ratio(
        train_validation_test_ratio=train_validation_test_ratio,
        max_num_elements=max_num_elements,
        use_masks=use_masks,
    )
    assert len(train_entries) > 0
    assert len(validation_entries) > 0
    assert len(test_entries) > 0

    train_split_odp = os.path.join(odp, "train", "raster", dataset_name)
    validation_split_odp = os.path.join(
        odp, "validation", "raster", dataset_name
    )
    test_split_odp = os.path.join(odp, "test", "raster", dataset_name)

    _copy_data_split(train_entries, train_split_odp)
    _copy_data_split(validation_entries, validation_split_odp)
    _copy_data_split(test_entries, test_split_odp)


def create_splitting_report(odp, **kwargs):
    ofp = os.path.join(odp, "splitting_report.txt")
    with open(ofp, "w") as report_file:
        for key in kwargs:
            report_file.write(f"{key}: {kwargs[key]}\n")
