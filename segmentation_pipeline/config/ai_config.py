import toml
from typing import List, Union, Tuple
from pydantic import BaseModel
from segmentation_pipeline.config.category_config import CategoryConfig

# https://pypi.org/project/dataclasses/
# https://github.com/samuelcolvin/pydantic/


class ModelConfig(BaseModel):
    segmentation_model_cfg_fp: str
    # NB: tile_size = (tile_width, tile_height)
    tile_size: Tuple[int, int]

    # For some panoptic segmentation models
    set_test_evaluator_outfile_prefix: Union[bool, int] = False


class AIConfig(BaseModel):
    categories: List[CategoryConfig]
    model: ModelConfig
    # auth: Optional[AuthentificationConfig]

    @classmethod
    def get_from_file(cls, toml_ifp):
        ai_config_dict = toml.load(toml_ifp)
        return cls(**ai_config_dict)