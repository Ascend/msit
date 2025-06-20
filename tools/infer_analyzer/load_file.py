import os
import sys
import stat

import logging
from constant import Constant
import pandas as pd

logger=logging.getLogger("infer_analyze")


class FileReader:
    def __init__(self, file_path):
        self.op_data = None
        self.batch_data = None
        self.json_path = None
        self.file_path = file_path

    def load_data(self):

        if os.path.isdir(os.path.join(self.file_path, "ASCEND_PROFILER_OUTPUT")):
            self.file_path = os.path.join(self.file_path, "ASCEND_PROFILER_OUTPUT")
        elif os.path.isdir(os.path.join(self.file_path, "mindstudio_profiler_output")):
            self.file_path = os.path.join(self.file_path, "mindstudio_profiler_output")
        else:
            logger.error("Wrong input path")
            sys.exit(0)
        for filename in os.listdir(self.file_path):
            if filename.startswith("op_summary"):
                op_path = os.path.join(self.file_path, filename)
                if FileOpen.check(op_path):
                    self.op_data = pd.read_csv(op_path, low_memory=False)
            if filename.startswith("kernel_details"):
                kernel_path = os.path.join(self.file_path, filename)
                if FileOpen.check(kernel_path):
                    self.op_data = pd.read_csv(kernel_path)
            if filename.startswith("msprof_tx") and filename.endswith("csv"):
                batch_path = os.path.join(self.file_path, filename)
                if FileOpen.check(batch_path):
                    self.batch_data = pd.read_csv(batch_path)
            if (filename.startswith("msprof_2") and filename.endswith("json")) or filename=="trace_view.json":
                json_path = os.path.join(self.file_path, filename)
                if FileOpen.check(json_path):
                    self.json_path = json_path
        if self.op_data is None or self.json_path is None:
            logger.error("Can't find all required file")
            sys.exit(0)



        return self.op_data,self.batch_data,self.json_path

class File:
    """
    open的安全文件操作类，使用with语句进行上下文管理
    """

    @staticmethod
    def check(path: str, max_size: int = Constant.MAX_FILE_BYTES) -> bool:
        """
        检查给定的文件路径是否有效。
        """
        if not path:
            logger.error("The path is empty. Please enter a valid path.")
            return False
        if len(path) > Constant.MAX_PATH_SIZE:
            logger.error(f"The length of file path is large than {Constant.MAX_PATH_SIZE}. Please check the path.")
            return False
        if os.path.getsize(path) > max_size:
            logger.error(f"The path \"{path}\" is too large to read. Please check the path.")
            return False
        if os.path.islink(path):
            logger.error(f"The path \"{path}\" is link. Please check the path.")
            return False
        return True

    @staticmethod
    def check_dir_for_create_file(file_dir):
        """
        创建文件需要目录有w和x权限，否则无法创建文件
        """
        if not os.access(file_dir, os.W_OK | os.X_OK):
            logger.error(f"The path \"{file_dir}\" does not have permission to create file. ")
            return False
        return True

    @staticmethod
    def create_file(file_path):
        """
        根据提供的路径创建一个文件，覆盖原有文件。如果目录不存在，则先创建目录。
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, mode=Constant.DIR_AUTHORITY)
        try:
            with open(file_path, 'w') as file:
                os.chmod(file_path, Constant.File_AUTHORITY)
        except Exception as e:
            logger.error(f"Failed to create file: {e}")

    @staticmethod
    def get_json_files(file_dir, end_type: str):
        for root, _, files in os.walk(file_dir):
            for file in files:
                if file.endswith(end_type):
                    yield os.path.join(root, file)


class FileOpen:
    """
    读取文件内容的类，使用with语句进行上下文管理
    """

    def __init__(self, path: str, mode: str = "r", max_size: int = Constant.MAX_FILE_BYTES):
        self.path = path
        self.mode = mode
        self.max_size = max_size
        self.file_reader = None

    def __enter__(self):
        if not self.check(self.path):
            logger.error(f"Cannot access the file: {self.path}")
            return None
        self.file_reader = open(self.path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_reader:
            self.file_reader.close()

    @staticmethod
    def check(file_path, max_size: int = Constant.MAX_FILE_BYTES):
        if not File.check(file_path, max_size):
            logger.error(f"FileReader check failed: {file_path}")
            return False
        if not os.path.isfile(file_path):
            logger.error(f"The read path \"{file_path}\" is not a file.")
            return False
        if not os.access(file_path, os.R_OK):
            logger.error(f"The path \"{file_path}\" does not have permission to read. ")
            return False
        return True


class FdOpen:
    """
    新建和写入文件内容的类，使用with语句进行上下文管理
    """

    def __init__(self, path: str, mode: str = "w", permission: int = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP,
                 flags: int = os.O_WRONLY | os.O_CREAT | os.O_TRUNC, newline: str = None) -> None:
        self.path = path
        self.mode = mode
        self.permission = permission
        self.flags = flags
        self.newline = newline
        self.fd = None
        self.file_open = None

    def __enter__(self):
        if not self.check(self.path):
            logger.error(f"Cannot access the file: {self.path}")
            return None
        self.fd = os.open(self.path, self.flags, self.permission)
        if self.newline is None:
            self.file_open = os.fdopen(self.fd, self.mode)
        else:
            self.file_open = os.fdopen(self.fd, self.mode, newline=self.newline)
        return self.file_open

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_open:
            self.file_open.close()
        elif self.fd:
            os.close(self.fd)

    @staticmethod
    def check(file_path, max_size: int = Constant.MAX_FILE_BYTES):
        if not os.path.exists(file_path):
            return File.check_dir_for_create_file(os.path.dirname(file_path))
        if not File.check(file_path, max_size):
            logger.error(f"FileReader check failed: {file_path}")
            return False
        if not os.path.isfile(file_path):
            logger.error(f"The write path \"{file_path}\" is not a file.")
            return False
        if not os.access(file_path, os.W_OK):
            logger.error(f"The path \"{file_path}\" does not have permission to write. ")
            return False
        return True
