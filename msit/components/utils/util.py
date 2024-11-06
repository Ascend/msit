# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os

from components.utils.constants import TENSOR_MAX_SIZE, EXT_SIZE_MAPPING


def get_entry_points(entry_points_name):
    try:
        from importlib import metadata

        return metadata.entry_points().get(entry_points_name, [])
    except Exception:
        import pkg_resources

        return list(pkg_resources.iter_entry_points(entry_points_name))


def confirmation_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)
    
    try:
        user_action = input(prompt)
    except Exception:
        return False
    
    return bool(confirm_pattern.match(user_action))


def check_file_ext(path, ext: str):
    if not isinstance(ext, str):
        raise TypeError("Expected type 'str', got %r instead" % type(exts))
    
    path_ext = os.path.splitext(path)[1]

    if path_ext != ext:
        return False
    
    return True


def check_file_size_based_on_ext(path, ext=None):
    """Check the file size based on extension. This function uses `os.stat` to get file size may lead to OSError"""

    if not isinstance(path, str):
        raise TypeError("Expected path to be 'str', got %r instead" % type(path))
    
    ext = ext or os.path.splitext(path)[1]
    size = os.path.getsize(path) # may lead to errors

    if ext in EXT_SIZE_MAPPING:
        if size > EXT_SIZE_MAPPING[ext]:
            return False
    else:
        if size > TENSOR_MAX_SIZE:
            confirmation_prompt = "The file %r is larger than expected. " \
                                "Attempting to read such a file could potentially impact system performance.\n" \
                                "Please confirm your awareness of the risks associated with this action ([y]/n): " % path
            return confirmation_interaction(confirmation_prompt)

    return True
