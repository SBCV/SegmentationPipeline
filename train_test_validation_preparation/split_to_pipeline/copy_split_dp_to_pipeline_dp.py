import os
import shutil

from eot.utility.os_ext import get_subdirs


def copy_split_data_to_pipeline_data(split_idp, pipeline_odp):
    """
    Input Structure:
        /isprs_potsdam_train_validation_test/test/raster/potsdam_0.7_0.1_0.2
        /isprs_potsdam_train_validation_test/train/raster/potsdam_0.7_0.1_0.2
        /isprs_potsdam_train_validation_test/validation/raster/potsdam_0.7_0.1_0.2

    Output Structure:
        /geo_data/test/data/potsdam_0.7_0.1_0.2
        /geo_data/train/data/potsdam_0.7_0.1_0.2
        /geo_data/validation/data/potsdam_0.7_0.1_0.2
    """

    data_type_idp_list = get_subdirs(split_idp, recursive=False)
    assert len(data_type_idp_list) == 3

    for data_type_idp in data_type_idp_list:
        # /test/raster
        raster_idp_list = get_subdirs(data_type_idp, recursive=False)
        assert len(raster_idp_list) == 1, raster_idp_list
        raster_idp = raster_idp_list[0]

        # /test/raster/potsdam_0.7_0.1_0.2
        dataset_idp_list = get_subdirs(raster_idp, recursive=False)
        assert len(dataset_idp_list) == 1
        dataset_idp = dataset_idp_list[0]

        dp_name = os.path.basename(data_type_idp)
        assert dp_name in ["train", "val", "test"]
        dataset_odp = os.path.join(pipeline_odp, dp_name, "data")

        shutil.copytree(raster_idp, dataset_odp)


if __name__ == "__main__":

    split_idp = "/mnt/DATA1-8TB/data_train_validation_test/isprs_potsdam_train_validation_test"
    pipeline_odp = "/mnt/DATA1-8TB/panoptic/plain_semantic_data_raster"
    dataset_name = "potsdam"
    copy_split_data_to_pipeline_data(split_idp, pipeline_odp)
