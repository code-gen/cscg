import json
from copy import deepcopy

class Config:
    """
    Fully manages a configuration
    """

    def __init__(self, data=None):
        if isinstance(data, str):
            self.from_json(data)
        if isinstance(data, dict):
            self.from_dict(data)

    def __str__(self):
        return self.to_json(indent=4)

    def __repr__(self):
        return str(self)

    @staticmethod
    def __get_nested(d):
        out = {}

        for k, v in d.items():
            # nested config
            if isinstance(v, Config):
                out[k] = Config.__get_nested(v.__dict__)

            # non-primitive type
            elif hasattr(v, '__dict__'):
                out[k] = str(v)

            # primitives
            else:
                out[k] = v

        return out

    @property
    def is_empty(self):
        return len(self.__dict__) == 0

    def to_json(self, indent=4) -> str:
        ns = self.__get_nested(self.__dict__)
        return json.dumps(ns, indent=indent)

    def to_file(self, fname) -> None:
        with open(fname, 'wt') as fp:
            fp.write(str(self))
            
    def copy(self):
        return deepcopy(self)