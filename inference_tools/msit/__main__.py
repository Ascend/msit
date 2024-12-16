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

from msit.common.cli import MainCommand
from msit.common.constants import CmdConst


def main():
    msit_command = MainCommand()
    msit_command.add_arguments()
    args = msit_command.parser.parse_args()
    if hasattr(args, CmdConst.RUN):
        args.run(args)
    else:
        msit_command.parser.print_help()


if __name__ == "__main__":
    main()
