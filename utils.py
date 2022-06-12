import pathlib

from typing import Union
from pathlib import Path


def check_path(path: Union[str, pathlib.Path], type='file'):
    if isinstance(path, str):
        path = Path(path)
    if type == 'file':
        if not path.is_file():
            raise FileNotFoundError(f'The file: {path} not found!')
    elif type == 'dir':
        if not path.is_dir():
            raise NotADirectoryError(f'The dir path: {path} not found!')

    return path

