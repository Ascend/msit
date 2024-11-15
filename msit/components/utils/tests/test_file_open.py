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
import os
import stat
import pytest
from components.utils.file_open_check import ms_open, FileStat, OpenException
from components.utils.file_open_check import PERMISSION_NORMAL, PERMISSION_KEY, RAW_INPUT_PATH
import tempfile
from pathlib import Path




@pytest.fixture(scope="function")
def not_exists_file_name():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_not_exists")
        if os.path.exists(file_name):
            os.remove(file_name)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_content_is_abcd():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_abcd")
        with ms_open(file_name, "w") as aa:
            aa.write("abcd")
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_permission_777():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_permission_777")
        with ms_open(file_name, "w") as aa:
            aa.write("abcd")
        os.chmod(file_name, 0o777)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_is_softlink():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_softlink")
        Path(f"{file_name}_src").touch()
        os.symlink(f"{file_name}_src", file_name)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


def test_msopen_given_mode_w_plus_when_write_4_lettle_then_file_writed_and_read_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "w+") as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "1234"
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_w_when_write_4_lettle_then_file_writed_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "w") as aa:
        aa.write("1234")

    assert FileStat(not_exists_file_name).file_size == 4
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_w_when_exists_file_and_write_4_lettle_then_file_writed_and_read_case(
    file_name_which_content_is_abcd,
):
    with ms_open(file_name_which_content_is_abcd, "w+") as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "1234"
    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner


def test_msopen_given_mode_x_when_write_4_lettle_then_file_writed_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "x") as aa:
        aa.write("1234")

    assert FileStat(not_exists_file_name).file_size == 4
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_x_when_exists_file_then_file_writed_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "x") as aa:
        aa.write("1234")


def test_msopen_given_mode_r_when_none_then_file_read_out_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "r", max_size=100) as aa:
        content = aa.read()
    assert content == "abcd"


def test_msopen_given_mode_r_plus_when_none_then_file_read_out_and_write_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "r+", max_size=100) as aa:
        content = aa.read()
        assert content == "abcd"
        aa.write("1234")


def test_msopen_given_mode_a_when_none_then_file_writed_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "a", max_size=100) as aa:
        aa.write("1234")

    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner

    with ms_open(file_name_which_content_is_abcd, "r", max_size=100) as aa:
        content = aa.read()
        assert content == "abcd1234"


def test_msopen_given_mode_a_plus_when_none_then_file_write_and_read_out_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "a+", max_size=100) as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "abcd1234"
    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner


def test_msopen_given_mode_r_when_file_not_exits_then_file_read_failed_case(not_exists_file_name):
    try:
        with ms_open(not_exists_file_name, "r", max_size=100) as aa:
            aa.read()
            assert False
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_r_no_max_length_when_none_then_file_read_failed_case(file_name_which_content_is_abcd):
    try:
        with ms_open(file_name_which_content_is_abcd, "r") as aa:
            assert False
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_r_max_size_2_when_none_then_file_failed_read_out_case(file_name_which_content_is_abcd):
    try:
        with ms_open(file_name_which_content_is_abcd, mode="r", max_size=3) as aa:
            assert False
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_w_when_file_permission_777_then_file_delete_before_write_case(
    file_name_which_permission_777,
):
    with ms_open(file_name_which_permission_777, mode="w") as aa:
        aa.write("1234")

    assert FileStat(file_name_which_permission_777).permission | PERMISSION_NORMAL == PERMISSION_NORMAL


def test_msopen_given_mode_a_when_file_permission_777_then_file_chmod_before_write_case(file_name_which_permission_777):
    with ms_open(file_name_which_permission_777, mode="a") as aa:
        aa.write("1234")

    assert FileStat(file_name_which_permission_777).permission | PERMISSION_NORMAL == PERMISSION_NORMAL


def test_msopen_given_mode_w_when_file_softlink_then_file_delete_before_write_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="w") as aa:
        aa.write("1234")

    assert FileStat(file_name_which_is_softlink).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert not FileStat(file_name_which_is_softlink).is_softlink


def test_msopen_given_mode_a_when_file_softlink_then_write_failed_case(file_name_which_is_softlink):
    try:
        with ms_open(file_name_which_is_softlink, mode="a") as aa:
            aa.write("1234")
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_w_p_600_when_file_softlink_then_file_delete_before_write_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="w", write_permission=PERMISSION_KEY) as aa:
        aa.write("1234")

    assert FileStat(file_name_which_is_softlink).permission | PERMISSION_KEY == PERMISSION_KEY


def test_msopen_given_mode_r_when_file_softlink_whitelist_empty_then_file_read_failed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = ""

    assert not FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_mode_r_when_file_softlink_target_right_then_file_read_succeed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = os.path.abspath(os.path.dirname(os.readlink(file_name_which_is_softlink)))

    assert FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_mode_r_when_file_softlink_target_wrong_then_file_read_failed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = "1234"

    assert not FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_other_w_parent_dir_then_file_read_failed_case():
    try:
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o702)
            fp = os.path.join(dp, "test_file")

            with ms_open(fp, mode="w") as aa:
                aa.write("no way")
    except OpenException as ignore:
        assert True
