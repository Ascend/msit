# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import logging.handlers


def create_logger(name: str, level: int = logging.DEBUG, use_memory_handler: bool = True) -> logging.Logger:
    """
    Create a logger with optional memory handler for buffering logs.

    Args:
        name: The name of the logger. recommend to use the module name: __name__.
        level: The logging level, default is DEBUG.
        use_memory_handler: Whether to add a memory handler for buffering logs, default is True.

    Returns:
        A configured logger instance.

    Examples:
        # Create a logger with memory handler
        logger = create_logger("my_logger", logging.INFO, use_memory_handler=True)

        # Create a logger without memory handler
        logger = create_logger("my_logger", logging.INFO, use_memory_handler=False)

    Notes:
        When use_memory_handler is True, a memory handler is added to buffer logs until a specific log level
        (default is ERROR) is reached, then logs are flushed to the target handler. This can avoid frequent
        file writes and improve performance. Buffered logs can be manually flushed by calling logger.handlers[1].flush()
        if no file handler is created yet.

        When use_memory_handler is False, no memory handler is added, and logs are written to the target handler
        (e.g., console or file) in real-time.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if use_memory_handler:
        memory_handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.ERROR)
        memory_handler.setLevel(level)
        memory_handler.setFormatter(formatter)
        logger.addHandler(memory_handler)

    return logger


def add_file_handler(logger: logging.Logger, log_file: str) -> logging.Logger:
    """
    Add a file handler to an existing logger and handle the memory handler if present.

    Args:
        logger: An existing logger instance.
        log_file: The path to the log file.

    Returns:
        The updated logger instance.

    Example:
        # Initialize a logger
        logger = create_logger("my_logger", logging.DEBUG, use_memory_handler=True)

        # Add a file handler to the logger
        logger = add_file_handler(logger, "output.log")

    Notes:
        This function adds a file handler to the given logger, inheriting the log level from the logger.
        If a memory handler was previously added to the logger, its target handler is set to the new file handler,
        buffered logs are flushed to the file, and then the memory handler is removed.
        This ensures that both buffered logs and subsequent logs are written to the file after using the file handler.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger.level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.MemoryHandler):
            handler.setTarget(file_handler)
            handler.flush()
            logger.removeHandler(handler)

    return logger


if __name__ == "__main__":
    logger = create_logger("test_logger", logging.DEBUG, use_memory_handler=True)
    logger.info("This is an info message from initial logger with memory handler")

    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name
        log = add_file_handler(logger, temp_file_path)
        logger.info("This is an info message from logger with file handler")
        logger.info("The log file is {}".format(temp_file_path))
