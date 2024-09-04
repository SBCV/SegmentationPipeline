from typing import List, Union
from pydantic import BaseModel

from mmseg_ext.categories.multi_dataset_category import (
    DatasetTypeToLabelValues,
)


class CategoryConfig(BaseModel):
    name: str
    is_ignore_category: bool = False
    palette_index: int
    palette_color: Union[str, List]
    is_thing: bool = False
    dataset_type_to_label_values: DatasetTypeToLabelValues
    weight: float = 1.0

