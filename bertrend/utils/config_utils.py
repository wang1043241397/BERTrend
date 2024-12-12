#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import ast
import json
import os
import tomllib
from configparser import BasicInterpolation
from pathlib import Path
from typing import Any


def _resolve_env_variables(config_dict: dict) -> dict:
    """
    Recursively resolve environment variables in a dictionary.
    Enviroment variables must be defined as a string like: $VAR_NAME.
    """
    for key, value in config_dict.items():
        # If the value is a string, check for environment variable
        if isinstance(value, str):
            config_dict[key] = os.path.expandvars(value)
        # If it's a nested dictionary, recurse
        elif isinstance(value, dict):
            config_dict[key] = _resolve_env_variables(value)
    return config_dict


def load_toml_config(config_file_path: Path) -> dict:
    """
    Load a TOML configuration file and resolve environment variables.
    """
    with open(config_file_path, "rb") as f:
        config_dict = tomllib.load(f)
    return _resolve_env_variables(config_dict)


def parse_literal(expr: Any) -> Any:
    """Allows to convert easily something like {'a': "2", 'b': "3", 3:'xyz', "c":"0.5", "z": "(1,1)"} into {'a': 2, 'b': 3, 3: 'xyz', 'c': 0.5, 'z': (1, 1)}"""
    if type(expr) is dict:
        return {k: parse_literal(v) for k, v in expr.items()}
    else:
        try:
            return ast.literal_eval(expr)
        except:
            return expr


class EnvInterpolation(BasicInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        value = super().before_get(parser, section, option, value, defaults)
        decoded_value = os.path.expandvars(value)
        try:
            # try to decode string as json
            return json.loads(decoded_value)
        except json.JSONDecodeError:
            # assumes string
            return decoded_value
