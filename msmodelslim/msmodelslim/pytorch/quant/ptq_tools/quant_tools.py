# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from __future__ import absolute_import, division, print_function
import os
import copy
import logging

import torch
import torch.nn as nn
import onnx
import numpy as np
from ascend_utils.common.security.pytorch import check_torch_module
from ascend_utils.common.security import check_type, get_valid_write_path, get_valid_read_path, safe_delete_path_if_exists, SafeWriteUmask
from ascend_utils.common import security
from msmodelslim.pytorch.quant.ptq_tools.quant_modules import Quantizer, Conv2dQuantizer, LinearQuantizer
from msmodelslim.pytorch.quant.ptq_tools.quant_deploy import quantize_model_deploy, convert_linear_params
from msmodelslim.pytorch.quant.ptq_tools.quant_deploy import ConvertLinearParams, ModelDeployQuantParams
from msmodelslim.pytorch.quant.ptq_tools.ptq_kia.quant_funcs import amp_decision  # squant algorithm api
from msmodelslim import logger
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig


class Calibrator(object):
    """ Calibrator for post-training quantization."""

    def __init__(self, model, cfg, calib_data=None, fuse_module_call_back=None):
        check_type(cfg, QuantConfig, param_name="cfg")
        check_torch_module(model)
        self.cfg = cfg
        if calib_data is None:
            calib_data = []
        self.calib_data = self.get_calib_data(calib_data)
        if not isinstance(calib_data, list):
            raise ValueError("calib_data should be a list of tensor data")

        if fuse_module_call_back is not None and callable(fuse_module_call_back):
            fuse_module_call_back(model)
        else:
            fuse_module(model)

        self.fp_model = copy.deepcopy(model)
        self.model = self.quantize(model)
        self.set_quant()

    def quantize(self, model):
        return quantize_model(model, cfg=self.cfg)

    def set_quant(self):
        self.enable_quant()
        if not self.cfg.act_quant:
            self.disable_input_quant()

        if not self.cfg.disable_names:
            set_first_last_layer(self.model)
        else:
            disable_quantization(self.model, self.cfg.disable_names)

    def enable_quant(self):
        enable_quantization(self.model)

    def disable_quant(self):
        disable_quantization(self.model)

    def disable_input_quant(self):
        disable_input_quantization(self.model)

    def get_calib_data(self, calib_data):
        if calib_data:
            if not isinstance(calib_data, list):
                raise ValueError("calib_data should be list of tensors")
            self.check_calib_data(calib_data)
            return calib_data

        if self.cfg.input_shape:
            try:
                rand_input = torch.rand(
                    self.cfg.input_shape, dtype=torch.float,
                    requires_grad=False
                )
            except RuntimeError as ex:
                logger.error("calib_data init failed, please check input_shape. %s", str(ex))
                raise ex
            calib_data = [[rand_input]]
            return calib_data

        raise ValueError("The calib_data or input_shape"
                         " should be offered")

    def check_calib_data(self, calib_data):
        for i, calib_data_item in enumerate(calib_data):
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            for _, item in enumerate(calib_data_item):
                if not isinstance(item, torch.Tensor):
                    raise ValueError("Not all elements in calib_data are torch.Tensor, "
                                     "please make sure that the model can run with model(*(calib_data[0]))")

    def amp(self, calib_amp=10):
        # for quantized model
        try:
            model_quant = copy.deepcopy(self.model)
        except BaseException as exception:
            model_quant = copy.deepcopy(self.fp_model)
            model_quant = self.quantize(model_quant)
            model_quant.eval()
            with torch.no_grad():
                for data in self.calib_data:
                    model_quant(*data)

        model_quant.eval()
        layer_names = list()
        out_feats = list()

        def hook(module, fea_in, fea_out):
            out_feats.append(fea_out)

        for name, module in model_quant.named_modules():
            if isinstance(module, Conv2dQuantizer) or\
                    isinstance(module, LinearQuantizer):
                layer_names.append(name)
                module.register_forward_hook(hook=hook)

        # for fp model
        model_fp = copy.deepcopy(self.fp_model)
        model_fp.eval()
        out_feats_fp = list()

        def hook_fp(module, fea_in, fea_out):
            out_feats_fp.append(fea_out)

        for name, module in model_fp.named_modules():
            if hasattr(nn.modules.linear, "NonDynamicallyQuantizableLinear"):
                is_linear = isinstance(module, nn.Linear) or \
                            isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear)
            else:
                is_linear = isinstance(module, nn.Linear)

            if isinstance(module, nn.Conv2d) or is_linear:
                if name in layer_names:
                    module.register_forward_hook(hook=hook_fp)


        fallback_layers = amp_decision(model_quant, model_fp,
                                       out_feats, out_feats_fp,
                                       layer_names, self.calib_data[:calib_amp], self.cfg.amp_num)
        if fallback_layers:
            logger.info("Fallback start")
            disable_quantization(self.model, names=fallback_layers)
            logger.info("Fallback end!")

    def run(self):
        try:
            self._run()
        except Exception as ex:
            raise Exception("Please check your config, model and input!", ex) from ex

    def get_quant_params(self):
        self.model.eval()
        input_scale = {}
        input_offset = {}
        weight_scale = {}
        weight_offset = {}
        quant_weight = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (Conv2dQuantizer, LinearQuantizer)):
                if not module.quant_weight.is_enable:
                    continue
                input_scale[name] = module.quant_input.input_scale
                input_offset[name] = module.quant_input.input_offset
                weight_scale[name] = module.quant_weight.weight_scale
                weight_offset[name] = module.quant_weight.weight_offset
                quant_weight[name] = module.quant_weight.int_weight_tensor

        return input_scale, input_offset, \
               weight_scale, weight_offset, quant_weight

    def save_param(self, output_path, output_name, output_file):
        output_path = os.path.join(output_path, output_name)
        output_path = get_valid_write_path(output_path)
        with SafeWriteUmask(umask=0o377):
            np.save(output_path, output_file)

    def export_param(self, save_path):
        logger.info("Path of quant param is %s ", save_path)
        if not os.path.exists(save_path):
            save_path = get_valid_write_path(save_path)
            os.makedirs(save_path)
        input_scale, input_offset, weight_scale, weight_offset, quant_weight = self.get_quant_params()
        self.save_param(save_path, "input_scale.npy", input_scale)
        self.save_param(save_path, "input_offset.npy", input_offset)
        self.save_param(save_path, "weight_scale.npy", weight_scale)
        self.save_param(save_path, "weight_offset.npy", weight_offset)
        self.save_param(save_path, "quant_weight.npy", quant_weight)
        logger.info("Save quant param success!")

    def export_onnx(self, model_arch, save_path, input_names):
        security.check_type(model_arch, str, param_name="model_arch")
        security.check_character(model_arch, param_name="model_arch")
        security.check_write_directory(save_path)
        security.check_element_type(input_names, str, list, param_name="input_names")
        security.check_character(input_names, param_name="input_names")

        if self.cfg.input_shape:
            input_shape = self.cfg.input_shape
            input_shape[0] = 1
            dummpy_input = torch.rand(
                input_shape, dtype=torch.float,
                requires_grad=False
            )
        else:
            dummpy_input = tuple(self.calib_data[0])
        self.fp_model.eval()

        if "yolov5" in model_arch.lower() :
            self.fp_model.model[-1].export = True

        temp_fp_model_file = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
        temp_fp_model_file = get_valid_write_path(temp_fp_model_file)

        with SafeWriteUmask():
            try:
                torch.onnx.export(self.fp_model, dummpy_input, temp_fp_model_file,
                                  input_names=input_names, verbose=True, opset_version=11)
            except RuntimeError as ex:
                logger.error("Export fp_model to onnx failed, please check model. %s", str(ex))
                raise ex

    def export_quant_onnx(self, model_arch, save_path, input_names=None, fuse_add=True, save_fp=False):
        security.check_type(model_arch, str, param_name="model_arch")
        security.check_character(model_arch, param_name="model_arch")
        security.check_write_directory(save_path)
        security.check_element_type(input_names, str, list, param_name="input_names")
        security.check_character(input_names, param_name="input_names")
        check_type(fuse_add, bool, param_name="fuse_add")
        check_type(save_fp, bool, param_name="save_fp")

        self.export_onnx(model_arch, save_path, input_names)
        onnx_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
        onnx_path = get_valid_read_path(onnx_path)
        model = onnx.load(onnx_path)

        if "swinv2" in model_arch.lower() or "solov2" in model_arch.lower():
            from onnxsim import simplify
            model, check = simplify(model)

        graph = model.graph
        nodes = graph.node

        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            self.get_quant_params()
        linear_params = ConvertLinearParams(
            onnx_model=model,
            input_scale=input_scale,
            input_offset=input_offset,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            quant_weight=quant_weight
        )
        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            convert_linear_params(linear_params)
        quantized_weight_namd = []
        for item in nodes:
            if item.op_type == "Conv":
                weight_name = ".".join(item.input[1].split(".")[:-1])
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("Conv, item.name :%s, weight_name :%s ", item.name, weight_name)

            elif item.op_type == "MatMul":
                weight_name = item.input[1]
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("MatMul, item.name :%s, weight_name :%s ", item.name, weight_name)

        quantize_model_deploy_params = ModelDeployQuantParams(
            quantized_weight_name=quantized_weight_namd, 
            quant_weight_dict=quant_weight,
            input_scale_dict=input_scale,
            input_offset_dict=input_offset,
            weight_scale_dict=weight_scale,
            weight_offset_dict=weight_offset,
            fuse_add=fuse_add
        )
        quantize_model_deploy(graph, quantize_model_deploy_params)

        temp_quant_model_file = os.path.join(save_path, "{}_quant.onnx".format(model_arch))
        temp_quant_model_file = get_valid_write_path(temp_quant_model_file)
        with SafeWriteUmask():
            onnx.save(model, temp_quant_model_file)
            logger.info("Quantification ended and onnx is stored in %s ", temp_quant_model_file)

        if not save_fp:
            save_fp_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
            safe_delete_path_if_exists(save_fp_path)

    def _run(self, calib_amp=10):
        logger.info("Calibration start!")
        self.model.eval()
        with torch.no_grad():
            for data in self.calib_data:
                self.model(*data)
        logger.info("Calibration end!")

        if self.cfg.amp_num > 0:
            logger.info("AMP start!")
            self.amp(calib_amp)
            logger.info("AMP end!")


def quantize_model(model, cfg=None):
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for sub_token in sub_tokens:
            cur_mod = getattr(cur_mod, sub_token)
        setattr(cur_mod, tokens[-1], module)

    for name, mod in model.named_modules():
        logger.info("quantize_model, name :%s, type of mod :%s ", name, type(mod))
        if isinstance(mod, nn.Conv2d):
            quant_mod = Conv2dQuantizer(cfg=cfg)
            quant_mod.set_param(mod)
            _set_module(model, name, quant_mod)
            continue

        if hasattr(nn.modules.linear, "NonDynamicallyQuantizableLinear"):
            is_linear = isinstance(mod, nn.Linear) or isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear)
        else:
            is_linear = isinstance(mod, nn.Linear)

        if is_linear:
            quant_mod = LinearQuantizer(cfg=cfg)
            quant_mod.set_param(mod)
            _set_module(model, name, quant_mod)
    return model


def set_first_last_layer(model, last=True):
    module_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, Conv2dQuantizer):
            logger.info("Quantized conv module:%s", name)
            module_list += [mod]
        if isinstance(mod, LinearQuantizer):
            logger.info("Quantized linear module:%s", name)
            module_list += [mod]
    module_list[0].quant_input.is_enable = False
    module_list[0].quant_weight.is_enable = False
    if last:
        module_list[-1].quant_weight.is_enable = False
        module_list[-1].quant_input.is_enable = False


def disable_input_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            logger.info("Disabling quantization of input quantizer:%s", name)
            module.disable_input_quantization()


def enable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            logger.info("Enabling quantizer:%s", name)
            module.enable_quantization(name)


def set_disable_quantization(module, name):
    module.disable_quantization(name)
    logger.info("Disabling quantizer:%s", name)


def disable_quantization(model, names=None):
    quantizer_names = list()
    if names is None:
        names = []
    if names:
        for name in names:
            quantizer_names.append(name + '.quant_input')
            quantizer_names.append(name + '.quant_weight')

    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            if (not quantizer_names) or (name in quantizer_names):
                set_disable_quantization(module, name)


def fuse_set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for sub_token in sub_tokens:
        cur_mod = getattr(cur_mod, sub_token)
    setattr(cur_mod, tokens[-1], module)


def fuse(conv, batchnorm):
    weight = conv.weight
    mean = batchnorm.running_mean
    var_sqrt = torch.sqrt(batchnorm.running_var + batchnorm.eps)

    beta = batchnorm.weight
    gamma = batchnorm.bias

    if conv.bias is not None:
        bias = conv.bias
    else:
        bias = mean.new_zeros(mean.shape)

    if (var_sqrt == 0).all():
        raise ZeroDivisionError
    else:
        weight = weight * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        bias = (bias - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           groups=conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(weight)
    fused_conv.bias = nn.Parameter(bias)
    return fused_conv


def fuse_conv_batchnorm_module(model, conv2d, conv2d_name, batchnorm, batchnorm_name):
    batchnorm_conv = fuse(conv2d, batchnorm)
    fuse_set_module(model, conv2d_name, batchnorm_conv)
    fuse_set_module(model, batchnorm_name, nn.Sequential())


def fuse_module(model):
    children = list(model.named_children())
    child_conv2d = None
    child_conv2d_name = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            if not isinstance(child_conv2d, torch.nn.Conv2d):
                pass
            elif child_conv2d.out_channels != child.running_mean.shape[0]:
                pass
            else:
                fuse_conv_batchnorm_module(model, child_conv2d, child_conv2d_name, child, name)
                child_conv2d = None
                child_conv2d_name = None
        elif isinstance(child, nn.Conv2d):
            child_conv2d = child
            child_conv2d_name = name
        else:
            fuse_module(child)
