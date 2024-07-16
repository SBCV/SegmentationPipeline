import toml
from pydantic import BaseModel


class ParsedBaseModel(BaseModel):
    @classmethod
    def get_from_file(cls, toml_ifp):
        config_dict = toml.load(toml_ifp)
        return cls(**config_dict)
