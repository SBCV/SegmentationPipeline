import copy
import os
from shutil import copyfile
from six.moves import urllib

from mmseg_ext.categories.multi_dataset_category import MultiDatasetCategory
from mmseg_ext.categories.multi_dataset_categories import (
    MultiDatasetCategories,
)


def create_config_from_template(config_template_ifp, config_fp, pipeline_config_cls):
    config_template_ifp = os.path.abspath(config_template_ifp)
    config_fp = os.path.abspath(config_fp)

    if not os.path.isfile(config_fp):
        copyfile(config_template_ifp, config_fp)
    return pipeline_config_cls.get_from_file(config_fp)


def configure_proxy_settings(proxy_config):

    if proxy_config.ip_str is not None:
        assert proxy_config.port_str is not None
        proxy = urllib.request.ProxyHandler(
            {"https": f"{proxy_config.ip_str}:{proxy_config.port_str}"}
        )
        opener = urllib.request.build_opener(proxy)
        # Install the openen on the module-level
        urllib.request.install_opener(opener)


def configure_multi_dataset_categories(pc):
    category_list = []
    for config_category in pc.ai_config.categories:
        config_category_vars = vars(config_category)
        category = MultiDatasetCategory(**config_category_vars)
        category.is_active = (
            category.name in pc.data_config.training_category_titles
        )
        category_list.append(category)
    multi_dataset_categories = MultiDatasetCategories(category_list)
    return multi_dataset_categories


def configure_tiling_scheme(pc, tiling_scheme):
    # fmt: off
    if tiling_scheme.represents_mercator_tiling():
        tiling_scheme.set_zoom_level(pc.tile_config.input_tile_zoom_level)
        train_tiling_scheme = copy.deepcopy(tiling_scheme)
        test_tiling_scheme = copy.deepcopy(tiling_scheme)
        validation_tiling_scheme = copy.deepcopy(tiling_scheme)
    elif tiling_scheme.represents_local_image_tiling():
        if tiling_scheme.is_in_pixel():
            tiling_scheme.set_tile_size_in_pixel(pc.tile_config.input_tile_size_in_pixel)
        if tiling_scheme.is_in_meter():
            tiling_scheme.set_tile_size_in_meter(pc.tile_config.input_tile_size_in_meter)

        train_tiling_scheme = copy.deepcopy(tiling_scheme)
        test_tiling_scheme = copy.deepcopy(tiling_scheme)
        validation_tiling_scheme = copy.deepcopy(tiling_scheme)

        train_tiling_scheme.set_alignment(pc.tile_config.input_train_tile_alignment)
        test_tiling_scheme.set_alignment(pc.tile_config.input_test_tile_alignment)
        # Use the test alignment for the validation tiling scheme
        validation_tiling_scheme.set_alignment(pc.tile_config.input_test_tile_alignment)

        train_tiling_scheme.set_overhanging_tiles_flag(pc.tile_config.input_train_tile_overhang)
        test_tiling_scheme.set_overhanging_tiles_flag(pc.tile_config.input_test_tile_overhang)
        # Use the test overhang for the validation tiling scheme
        validation_tiling_scheme.set_overhanging_tiles_flag(pc.tile_config.input_test_tile_overhang)

        train_tiling_scheme.set_border_tiles_flag(pc.tile_config.input_train_tile_keep_border_tiles)
        test_tiling_scheme.set_border_tiles_flag(pc.tile_config.input_test_tile_keep_border_tiles)
        # Use the test border tile flag for the validation tiling scheme
        validation_tiling_scheme.set_border_tiles_flag(pc.tile_config.input_test_tile_keep_border_tiles)

        if tiling_scheme.is_in_pixel():
            train_tiling_scheme.set_tile_stride_in_pixel(pc.tile_config.input_train_tile_stride_in_pixel)
            test_tiling_scheme.set_tile_stride_in_pixel(pc.tile_config.input_test_tile_stride_in_pixel)
            # Use the tile size as stride
            validation_tiling_scheme.set_tile_stride_in_pixel(pc.tile_config.input_tile_size_in_pixel)
        if tiling_scheme.is_in_meter():
            train_tiling_scheme.set_tile_stride_in_meter(pc.tile_config.input_train_tile_stride_in_meter)
            test_tiling_scheme.set_tile_stride_in_meter(pc.tile_config.input_test_tile_stride_in_meter)
            # Use the tile size as stride
            validation_tiling_scheme.set_tile_stride_in_meter(pc.tile_config.input_tile_size_in_meter)
    else:
        assert False
    # fmt: on
    return train_tiling_scheme, test_tiling_scheme, validation_tiling_scheme
