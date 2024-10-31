import os
import stat
from unittest.mock import patch
import pytest

from msit_llm.opcheck.opchecker import _is_atb_only_saved_before, OpChecker
from msit_llm.common.log import logger

@pytest.fixture()
def mock_logger():
    with patch.object(logger, 'error') as mock_error:
        yield mock_error


@pytest.mark.parametrize(
    "dump_scenario, expected_result",
    [
        (['before'], True),
        (['after'], False),
        (['before', 'after'], False)
    ]
)
def test_is_atb_only_saved_before(mock_logger, tmpdir, dump_scenario, expected_result):
    input_path = tmpdir.mkdir('token_id')
    layer_dir = input_path.mkdir('0_Decoder_layer')
    for folders in dump_scenario:
        layer_dir.join(folders).ensure()
    res = _is_atb_only_saved_before(str(input_path))

    assert res is expected_result
    mock_logger.assert_not_called()


def test_is_atb_only_saved_before_false_no_folders(mock_logger, tmpdir):
    input_path = tmpdir.mkdir('token_id')
    res = _is_atb_only_saved_before(str(input_path))

    assert res is False
    mock_logger.assert_called_once()


def third_party_init_env_path(tmp_path_factory):
    # 预置so加载环境变量
    dir_path = tmp_path_factory.mktemp("test_ait_opcheck_lib_path")
    file_path = os.path.join( dir_path / "libopchecker.so")
    os.environ['AIT_OPCHECK_LIB_PATH'] = file_path
    return file_path

def test_third_party_init_load_not_exist_file(tmp_path_factory):
    third_party_init_env_path(tmp_path_factory)

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    assert res is False


def test_third_party_init_load_other_writable_file(tmp_path_factory):
    file_path = third_party_init_env_path(tmp_path_factory)

    # 创建他人可写文件
    file_permissions = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    with os.fdopen(os.open(file_path, os.O_CREAT, file_permissions), 'w'):
        pass

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    if os.path.exists(file_path):
        os.remove(file_path)
    assert res is False