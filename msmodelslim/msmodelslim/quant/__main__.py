#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse

from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig, W8A8ProcessorConfig
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session import quant_model, SessionConfig
from msmodelslim.quant.session.plugin import load_plugin

DEFAULT_CFG_MAP = {
    "w8a8": W8A8QuantConfig(
        w_cfg=dict(
            bits=8,
            symmetric=True,
            per_channel=True
        ),
        a_cfg=dict(
            bits=8,
            symmetric=True,
            per_channel=False
        )
    )
}


def main():
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("-p", "--plugin_path", type=str, help="插件文件路径")
    parser.add_argument("-t", "--quant_type", type=str, help="量化类型", default="w8a8")
    args = parser.parse_args()

    plugin = load_plugin(args.plugin_path, args)

    default_linear_config = DEFAULT_CFG_MAP.get(args.quant_type, None)

    default_cfg_map = {
        "*": default_linear_config
    }

    default_save_cfg = SaverProcessorConfig(
        save_output_path=".",
        safetensors_name="w8a8_model.safetensors",
        json_name="w8a8_config.json",
        save_type="safe_tensor"
    )

    default_cfg = W8A8ProcessorConfig(
        cfg_map=default_cfg_map
    )

    model = plugin.load_model()
    calib_data = plugin.load_calib_data()
    quant_cfg = plugin.load_quant_cfg(default_cfg)
    save_cfg = plugin.get_save_cfg(default_save_cfg)

    session_config = SessionConfig(
        processor_cfg_map={
            args.quant_type: quant_cfg,
            "save": save_cfg
        },
        calib_data=calib_data
    )

    session_config.model_validate(session_config)

    quant_model(model, session_config)


if __name__ == "__main__":
    main()
