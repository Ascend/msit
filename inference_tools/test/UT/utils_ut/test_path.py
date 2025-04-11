import os
import tempfile
import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from msit.utils.constants import PathConst
from msit.utils.exceptions import MsitException
from msit.utils.path import (
    _MAX_DIR_DEPTH,
    _MAX_LAST_NAME_LENGTH,
    _MAX_PATH_LENGTH,
    _MODE,
    AUTHORITY_DIR,
    SOFT_LINK_LEVEL_IGNORE,
    SOFT_LINK_LEVEL_STRICT,
    SOFT_LINK_LEVEL_WARNING,
    MsitPath,
    change_permission,
    convert_bytes,
    get_basename_from_path,
    get_dir_size,
    get_file_size,
    get_name_and_ext,
    is_dir,
    is_enough_disk_space,
    is_file,
    is_saved_model_scene,
    join_path,
    make_dir,
)


class TestIsFile(unittest.TestCase):
    def test_existing_file(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(is_file(tmp.name))

    def test_non_existing_path(self):
        self.assertFalse(is_file("/non/existent/path"))

    def test_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_file(tmpdir))


class TestIsDir(unittest.TestCase):
    def test_existing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(is_dir(tmpdir))

    def test_non_existing_path(self):
        self.assertFalse(is_dir("/non/existent/path"))

    def test_file(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.assertFalse(is_dir(tmp.name))


class TestGetBasenameFromPath(unittest.TestCase):
    def test_normal_path(self):
        self.assertEqual(get_basename_from_path("/path/to/file.txt"), "file.txt")

    def test_trailing_slash(self):
        self.assertEqual(get_basename_from_path("/path/to/dir/"), "dir")

    def test_root_path(self):
        self.assertEqual(get_basename_from_path("/"), "")


class TestGetFileSize(unittest.TestCase):
    def test_file_size(self):
        with tempfile.NamedTemporaryFile() as tmp:
            content = b"12345"
            tmp.write(content)
            tmp.flush()
            self.assertEqual(get_file_size(tmp.name), len(content))

    def test_non_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            get_file_size("/invalid/path")


class TestGetNameAndExt(unittest.TestCase):
    def test_with_extension(self):
        self.assertEqual(get_name_and_ext("/path/to/file.txt"), ("file", ".txt"))

    def test_multiple_dots(self):
        self.assertEqual(get_name_and_ext("file.tar.gz"), ("file.tar", ".gz"))

    def test_no_extension(self):
        self.assertEqual(get_name_and_ext("/path/to/file"), ("file", ""))


class TestJoinPath(unittest.TestCase):
    def test_basic_join(self):
        self.assertEqual(join_path("a", "b", "c"), os.path.join("a", "b", "c"))

    def test_nested_iterables(self):
        self.assertEqual(join_path(["a", ["b", "c"]], "d"), os.path.join("a", "b", "c", "d"))

    def test_max_depth_exceeded(self):
        deep_nested = ["a", ["b", ["c"]]]
        with self.assertRaises(MsitException) as e:
            join_path(deep_nested, max_depth=2)
        self.assertIn("Maximum recursion depth 2 exceeded", str(e.exception))

    def test_invalid_max_depth_type(self):
        with self.assertRaises(MsitException) as e:
            join_path("a", max_depth="invalid")
        self.assertIn("max_depth must be a positive integer.", str(e.exception))


class TestIsSavedModelScene(unittest.TestCase):
    def create_valid_structure(self, path):
        os.makedirs(os.path.join(path, "variables"))
        with open(os.path.join(path, "saved_model.pb"), "w") as f:
            f.write("")

    def test_valid_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_valid_structure(tmpdir)
            self.assertTrue(is_saved_model_scene(tmpdir))

    def test_missing_pb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "variables"))
            self.assertFalse(is_saved_model_scene(tmpdir))

    def test_missing_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "saved_model.pb"), "w") as f:
                f.write("")
            self.assertFalse(is_saved_model_scene(tmpdir))


class TestConvertBytes(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(convert_bytes(500), "500 Bytes")

    def test_kb(self):
        self.assertEqual(convert_bytes(2048), "2.00 KB")

    def test_mb(self):
        self.assertEqual(convert_bytes(2 * 1024 * 1024), "2.00 MB")

    def test_gb(self):
        self.assertEqual(convert_bytes(3 * 1024 * 1024 * 1024), "3.00 GB")

    def test_zero(self):
        self.assertEqual(convert_bytes(0), "0 Bytes")


class TestMsitPathInitialization(unittest.TestCase):
    def test_valid_initialization(self):
        msit_path = MsitPath(
            path="/tmp/test", path_type=PathConst.FILE, mode="r", size_limitation=1024, suffix=".txt", max_dir_depth=10
        )
        self.assertEqual(msit_path.path, "/tmp/test")

    def test_invalid_path_type(self):
        with self.assertRaisesRegex(MsitException, "path type must be one of") as e:
            MsitPath("/tmp", "invalid_type", "r")
        self.assertIn("The path type must be one of", str(e.exception))

    def test_invalid_mode(self):
        with self.assertRaisesRegex(MsitException, "Mode must be one of") as e:
            MsitPath("/tmp", PathConst.DIR, "invalid_mode")
        self.assertIn("Mode must be one of ['r', 'rb', 'w', 'wb', 'a', 'ab', 'a+', 'e']", str(e.exception))

    def test_negative_size_limitation(self):
        with self.assertRaisesRegex(MsitException, "greater than 0") as e:
            MsitPath("/tmp", PathConst.FILE, "r", size_limitation=-1)
        self.assertIn("The value must be an integer greater than 0, currently: -1.", str(e.exception))


class TestMsitPath(unittest.TestCase):
    def test_check_path_type_valid(self):
        self.assertEqual(MsitPath._check_path_type(PathConst.FILE), PathConst.FILE)
        self.assertEqual(MsitPath._check_path_type(PathConst.DIR), PathConst.DIR)

    def test_check_path_type_invalid(self):
        with self.assertRaises(MsitException) as e:
            MsitPath._check_path_type("invalid_type")
        self.assertIn("The path type must be one of ", str(e.exception))

    def test_check_mode_valid(self):
        for mode in _MODE:
            self.assertEqual(MsitPath._check_mode(mode), mode)

    def test_check_mode_invalid(self):
        with self.assertRaises(MsitException):
            MsitPath._check_mode("invalid_mode")

    def test_check_positive_int_valid(self):
        self.assertEqual(MsitPath._check_positive_int(5), 5)

    def test_check_positive_int_invalid(self):
        with self.assertRaises(MsitException):
            MsitPath._check_positive_int(0)
        with self.assertRaises(MsitException):
            MsitPath._check_positive_int(-1)
        with self.assertRaises(MsitException):
            MsitPath._check_positive_int("not_int")

    @patch("msit.utils.path.os.path.getsize")
    @patch("msit.utils.path.os.path.exists")
    @patch("msit.utils.path.os.path.islink")
    @patch("msit.utils.path.os.path.realpath")
    @patch("msit.utils.path.os.stat")
    @patch("msit.utils.path.is_file")
    @patch("msit.utils.path.is_dir")
    def test_check_existing_file(
        self, mock_is_dir, mock_is_file, mock_stat, mock_realpath, mock_islink, mock_exists, mock_getsize
    ):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_is_dir.return_value = False
        mock_islink.return_value = False
        mock_realpath.return_value = "/valid/path/file.txt"
        mock_getsize.return_value = 512
        stat_mock = MagicMock()
        stat_mock.st_uid = 0
        stat_mock.st_mode = 0o755
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/valid/path/file.txt", PathConst.FILE, "r", size_limitation=1024, suffix=".txt")
        result = msit_path.check()
        self.assertEqual(result, "/valid/path/file.txt")
        mock_getsize.assert_called_once_with("/valid/path/file.txt")

    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.realpath")
    @patch("msit.utils.path.is_dir")
    @patch("os.stat")
    def test_check_write_mode_new_file(self, mock_stat, mock_is_dir, mock_realpath, mock_islink, mock_exists):
        mock_exists.side_effect = lambda x: False if x == "/new/file.txt" else True
        mock_is_dir.return_value = True
        mock_islink.return_value = False
        mock_realpath.return_value = "/valid/parent"

        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = os.geteuid()
        mock_stat_result.st_mode = 0o755
        mock_stat.return_value = mock_stat_result

        msit_path = MsitPath("/new/file.txt", PathConst.FILE, "w")
        result = msit_path.check(path_exist=False)
        self.assertTrue(result.endswith("/new/file.txt"))

    @patch("os.path.abspath")
    @patch("os.path.normpath")
    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.realpath")
    @patch("msit.utils.path.is_file")
    @patch("os.stat")
    @patch("os.path.getsize")
    def test_soft_link_validation(
        self,
        mock_getsize,
        mock_stat,
        mock_is_file,
        mock_realpath,
        mock_islink,
        mock_exists,
        mock_normpath,
        mock_abspath,
    ):
        mock_abspath.side_effect = lambda x: x
        mock_normpath.side_effect = lambda x: x

        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = os.geteuid()
        mock_stat_result.st_mode = 0o755
        mock_stat.return_value = mock_stat_result

        mock_is_file.return_value = True
        mock_exists.return_value = True
        mock_islink.return_value = True
        mock_realpath.return_value = "/real/path"
        mock_getsize.return_value = 512
        msit_path = MsitPath("/symlink/path", PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check(soft_link_level=SOFT_LINK_LEVEL_STRICT)
        self.assertIn("is a symlink. Usage prohibited.", str(e.exception))
        mock_islink.assert_called_with("/symlink/path")
        mock_realpath.assert_called_with("/symlink/path")

    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.realpath")
    def test_soft_link_non_validation(self, mock_realpath, mock_islink, mock_exists):
        mock_exists.return_value = True
        mock_islink.return_value = True
        mock_realpath.return_value = "/real/path"
        msit_path = MsitPath("/symlink/path", PathConst.FILE, "r")
        with self.assertRaises(MsitException) as context:
            msit_path.check(soft_link_level=4)
        self.assertIn("The validation level of symbolic links must be one of ", str(context.exception))

    @patch("msit.utils.path.is_file")
    @patch("msit.utils.path.is_dir")
    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.realpath")
    @patch("os.stat")
    def test_soft_link_ignore_validation(
        self, mock_stat, mock_realpath, mock_islink, mock_exists, mock_is_dir, mock_is_file
    ):
        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = os.geteuid()
        mock_stat_result.st_mode = 0o755
        mock_stat.return_value = mock_stat_result
        mock_exists.return_value = True
        mock_islink.return_value = True
        mock_realpath.return_value = "/real/path"
        mock_is_file.return_value = True
        mock_is_dir.return_value = False
        msit_path = MsitPath("/symlink/path", PathConst.FILE, "r")
        result = msit_path.check(soft_link_level=SOFT_LINK_LEVEL_IGNORE)
        self.assertEqual(result, "/real/path")
        mock_is_file.assert_called_with("/real/path")

    @patch("msit.utils.path.is_file")
    @patch("os.path.exists")
    @patch("os.path.islink")
    @patch("os.path.realpath")
    @patch("os.stat")
    def test_path_length_validation(self, mock_stat, mock_realpath, mock_islink, mock_exists, mock_is_file):
        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = os.geteuid()
        mock_stat_result.st_mode = 0o755
        mock_stat.return_value = mock_stat_result
        mock_realpath.return_value = "/real/path"
        mock_islink.return_value = False
        mock_exists.return_value = True
        mock_is_file.return_value = True
        long_path = "/" + "a" * (_MAX_PATH_LENGTH + 1)
        msit_path = MsitPath(long_path, PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check(path_exist=False)
        self.assertIn("Current path length (4098) exceeds the limit (4096).", str(e.exception))

    @patch("msit.utils.path.is_file")
    @patch("os.stat")
    @patch("os.geteuid")
    def test_permission_validation(self, mock_geteuid, mock_stat, mock_is_file):
        mock_geteuid.return_value = 1000
        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = 1000
        mock_stat_result.st_mode = 0o777
        mock_stat.return_value = mock_stat_result
        mock_is_file.return_value = True
        msit_path = MsitPath("/unsafe/path", PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("Permissions for files (or directories) should not exceed 0o755 (rwxr-xr-x).", str(e.exception))

    @patch("msit.utils.path.is_file")
    @patch("os.stat")
    @patch("os.geteuid")
    def test_permission_validation(self, mock_geteuid, mock_stat, mock_is_file):
        mock_geteuid.return_value = 1000
        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = 500
        mock_stat_result.st_mode = 0o777
        mock_stat.return_value = mock_stat_result
        mock_is_file.return_value = True
        msit_path = MsitPath("/unsafe/path", PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("The owner of /unsafe/path must be root or the current user.", str(e.exception))

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_read_permission_denied(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o300
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/no_read.txt", PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("not authorized to read", str(e.exception))

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_read_permission_granted(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o400
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/readable.txt", PathConst.FILE, "r")
        msit_path.check()

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_write_permission_denied(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o500
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/no_write.txt", PathConst.FILE, "w")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("not authorized to write", str(e.exception))

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_write_permission_granted(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o600
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/writable.txt", PathConst.FILE, "w")
        msit_path.check()

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_execute_permission_denied(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o600
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/no_execute.txt", PathConst.FILE, "e")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("not authorized to execute", str(e.exception))

    @patch("os.stat")
    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_execute_permission_granted(self, mock_is_file, mock_exists, mock_stat):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o500
        mock_stat.return_value = stat_mock
        msit_path = MsitPath("/executable.txt", PathConst.FILE, "e")
        msit_path.check()

    @patch("msit.utils.path.is_dir")
    @patch("os.path.exists")
    @patch("os.stat")
    def test_directory_validation(self, mock_stat, mock_exists, mock_is_dir):
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o750
        mock_stat.return_value = stat_mock
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        msit_path = MsitPath("/valid/dir", PathConst.DIR, "r")
        msit_path.check()
        mock_is_dir.return_value = False
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("is not a directory", str(e.exception))

    @patch("msit.utils.path.is_dir")
    @patch("os.path.exists")
    @patch("os.stat")
    def test_special_char_validation(self, mock_stat, mock_exists, mock_is_dir):
        stat_mock = MagicMock()
        stat_mock.st_uid = os.geteuid()
        stat_mock.st_mode = 0o750
        mock_stat.return_value = stat_mock
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        msit_path = MsitPath("/valid/123%", PathConst.DIR, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("Path /valid/123% contains special characters.", str(e.exception))

    @patch("os.path.exists")
    @patch("msit.utils.path.is_dir")
    def test_directory_depth_validation(self, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        over_depth_path = "/level1/" + "/".join([f"level{i}" for i in range(2, _MAX_DIR_DEPTH + 2)])
        msit_path = MsitPath(over_depth_path, PathConst.DIR, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn(f"Exceeded max directory depth ({_MAX_DIR_DEPTH})", str(e.exception))

    @patch("os.path.exists")
    @patch("msit.utils.path.is_file")
    def test_filename_length_validation(self, mock_is_file, mock_exists):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        long_name = "a" * (_MAX_LAST_NAME_LENGTH + 1)
        invalid_path = f"/normal_dir/{long_name}"
        msit_path = MsitPath(invalid_path, PathConst.FILE, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn(f"length ({_MAX_LAST_NAME_LENGTH + 1}) exceeds", str(e.exception))

    @patch("os.path.exists")
    @patch("msit.utils.path.is_dir")
    def test_multiple_long_directory_names(self, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        long_dir = "b" * (_MAX_LAST_NAME_LENGTH + 1)
        invalid_path = f"/{long_dir}/{long_dir}"
        msit_path = MsitPath(invalid_path, PathConst.DIR, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertEqual(str(e.exception).count("exceeds the limit"), 1)

    @patch("os.path.exists")
    @patch("msit.utils.path.is_dir")
    def test_mixed_error_conditions(self, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        long_dir = "c" * (_MAX_LAST_NAME_LENGTH + 1)
        deep_path = "/" + "/".join([long_dir] * (_MAX_DIR_DEPTH + 2))
        msit_path = MsitPath(deep_path, PathConst.DIR, "r")
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("Current path length (8738) exceeds the limit (4096).", str(e.exception))

    @patch("os.path.exists")
    @patch("msit.utils.path.is_dir")
    @patch("msit.utils.path.get_dir_size")
    def test_directory_size_validation(self, mock_get_dir_size, mock_is_dir, mock_exists):
        mock_get_dir_size.return_value = 1024 * 1024 * 1024
        mock_is_dir.return_value = True
        mock_exists.return_value = True
        msit_path = MsitPath("/large/dir", PathConst.DIR, "r", size_limitation=100)
        with self.assertRaises(MsitException) as e:
            msit_path.check()
        self.assertIn("Directory size exceeds the limit (100 Bytes).", str(e.exception))


class TestGetDirSize(unittest.TestCase):
    @patch("os.walk")
    @patch("os.path.getsize")
    def test_get_dir_size_success(self, mock_getsize, mock_walk):
        dir_path = "/test"
        mock_walk.return_value = [
            (dir_path, ["sub1"], ["file1", "file2"]),
            (os.path.join(dir_path, "sub1"), [], ["file3"]),
        ]
        mock_getsize.side_effect = [100, 200, 300]
        result = get_dir_size(dir_path, max_dir_depth=2)
        self.assertEqual(result, 600)

    @patch("os.walk")
    def test_get_dir_size_exceed_max_depth(self, mock_walk):
        dir_path = "/test"
        mock_walk.return_value = [
            (dir_path, ["sub1"], []),
            (os.path.join(dir_path, "sub1"), ["sub2"], []),
            (os.path.join(dir_path, "sub1", "sub2"), [], ["file"]),
        ]
        with self.assertRaises(MsitException) as cm:
            get_dir_size(dir_path, max_dir_depth=1)
        self.assertIn("exceeded max depth (1)", str(cm.exception))


class TestMakeDir(unittest.TestCase):
    @patch("msit.utils.path.Path")
    @patch("msit.utils.path.MsitPath")
    def test_make_dir_success(self, mock_msitpath, mock_path):
        mock_msit_instance = MagicMock()
        mock_msitpath.return_value = mock_msit_instance
        mock_msit_instance.check.return_value = "/valid/dir"
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        make_dir("test_dir")
        mock_msitpath.assert_called_once_with("test_dir", PathConst.DIR, "w")
        mock_msit_instance.check.assert_called_once_with(path_exist=False)
        mock_path.assert_called_once_with("/valid/dir")
        mock_path_instance.mkdir.assert_called_once_with(mode=AUTHORITY_DIR, exist_ok=True, parents=False)

    @patch("msit.utils.path.Path")
    @patch("msit.utils.path.MsitPath")
    def test_make_dir_oserror_parents(self, mock_msitpath, mock_path):
        mock_instance = MagicMock()
        mock_msitpath.return_value = mock_instance
        mock_instance.check.return_value = "/invalid/parent_dir"
        mock_path.return_value.mkdir.side_effect = OSError("Parent missing")
        with self.assertRaises(MsitException) as cm:
            make_dir("bad_dir")
        self.assertIn("Check if the parent directory", str(cm.exception))


class TestChangePermission(unittest.TestCase):
    @patch("os.chmod")
    @patch("os.path.islink")
    @patch("os.path.exists")
    def test_change_permission_success(self, mock_exists, mock_islink, mock_chmod):
        mock_exists.return_value = True
        mock_islink.return_value = False
        change_permission("/valid/file", 0o755)
        mock_chmod.assert_called_once_with("/valid/file", 0o755)

    @patch("os.chmod")
    @patch("os.path.islink")
    @patch("os.path.exists")
    def test_change_permission_skip_symlink(self, mock_exists, mock_islink, mock_chmod):
        mock_exists.return_value = True
        mock_islink.return_value = True
        change_permission("/symlink", 0o755)
        mock_chmod.assert_not_called()

    @patch("os.chmod")
    @patch("os.path.islink")
    @patch("os.path.exists")
    def test_change_permission_permission_error(self, mock_exists, mock_islink, mock_chmod):
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_chmod.side_effect = PermissionError("Permission denied")
        with self.assertRaises(MsitException) as cm:
            change_permission("/restricted/file", 0o777)
        self.assertIn("Failed to set permissions (511)", str(cm.exception))
        self.assertIn("/restricted/file", str(cm.exception))


class TestDiskSpaceCheck(unittest.TestCase):
    @patch("msit.utils.path.disk_usage")
    def test_not_enough_space_returns_true(self, mock_disk_usage: MagicMock):
        mock_result = MagicMock()
        mock_result.free = 1000
        mock_disk_usage.return_value = mock_result
        result = is_enough_disk_space("/test/path", 500)
        self.assertTrue(result)
        mock_disk_usage.assert_called_once_with("/test/path")

    @patch("msit.utils.path.disk_usage")
    def test_enough_space_returns_false(self, mock_disk_usage: MagicMock):
        mock_result = MagicMock()
        mock_result.free = 1000
        mock_disk_usage.return_value = mock_result
        result = is_enough_disk_space("/test/path", 1500)
        self.assertFalse(result)

    @patch("msit.utils.path.disk_usage")
    def test_exact_space_edge_case(self, mock_disk_usage: MagicMock):
        mock_result = MagicMock()
        mock_result.free = 1000
        mock_disk_usage.return_value = mock_result
        result = is_enough_disk_space("/test/path", 1000)
        self.assertTrue(result)
