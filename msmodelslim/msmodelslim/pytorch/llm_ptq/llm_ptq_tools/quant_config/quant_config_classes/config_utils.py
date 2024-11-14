# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from ascend_utils.common.security import check_number, check_type, check_element_type
from ascend_utils.common.security.pytorch import validate_device
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, WeightQuantMethod

_SUPPORTED_DEVICES = ["cpu", "npu", 'gpu']
A_BIT_LIST = [8, 16]
W_BIT_LIST = [4, 8]
GROUP_SIZE_LIST = [64, 128]


def set_quant_param(config):
    config.w_signed = True
    config.a_signed = True
    config.a_sym = False
    config.input_shape = []
    config.act_quant = True
    config.amp_num = 0
    config.keep_acc = {'admm': [False, 1000], 'round_opt': False}
    config.hessian_optim = {'flag': False, 'std_threshold': -1, 'sigma_weight': 5}
    config.seq_opt = {'en': False, 'n': 10, 'bin': 0.01}
    config.int_bias = False
    config.calib_mode = 0
    config.model_quant_type = QuantType.UNKNOWN


def set_lowbit_param(config):
    config.use_amp = False
    config.optimize_input_amax = False
    config.smooth_down = False
    config.ch_align = True
    config.search_smooth = True
    config.default_smooth_alpha = 0.5
    config.search_intervals = 10
    config.quant_down = True
    config.offload = False
    config.lwc = False
    config.lac = False
    config.search_omniquant = False
    config.use_low_rank = False
    config.optimizate_msd = False
    config.fold = 3
    config.do_msd = False
    config.down_proj_type = ['c_proj', 'down_proj', 'dense_4h_to_h']
    config.norm_class_name = 'RMSNorm'


def set_per_group_param(config):
    is_per_group = config.a_bit == 16 and not config.open_outlier and config.is_lowbit
    if is_per_group:
        if config.group_size not in GROUP_SIZE_LIST:
            raise ValueError(f"group_size must be among choice {GROUP_SIZE_LIST}, please check it.")
        config.do_msd = True
        config.fold = 3
        config.optimizate_msd = True
        config.do_smooth = False
        config.amp_num = 0
        config.mm_tensor = False
    else:
        config.group_size = -1
        if config.is_lowbit and (config.w_method == 'HQQ' or config.w_method == 'GPTQ'):
            raise ValueError("When is_lowbit is set to True, the w_method configuration is only supported in a "
                             "per-group scenario. Please check the config.")


def set_fa_quant_param(config):
    if not hasattr(config, "use_fa_quant"):
        config.use_fa_quant = False
    if not hasattr(config, "fa_amp"):
        config.fa_amp = 0
    if not hasattr(config, "fa_tp_size"):
        config.fa_tp_size = 1


def check_dynamic_config(config):
    check_type(config.is_dynamic, bool, param_name='is_dynamic')
    if config.is_dynamic:
        if config.w_bit != 8:
            raise ValueError("w_bit must be 8 when running dynamic quantization.")
        if config.a_bit != 8:
            raise ValueError("a_bit must be 8 when running dynamic quantization.")


def check_sparse_config(config):
    check_type(config.nonuniform, bool, param_name="nonuniform")
    check_number(config.fraction, float, 0.01, 0.1, param_name="fraction")

    if config.co_sparse and config.is_lowbit:
        config.co_sparse = False

    if config.co_sparse:
        config.pr = 1.0
        if config.w_bit != 4:
            config.w_bit = 4
            msmodelslim_logger.warning("Running in sparse requires `w_bit` of 4, "
                                       "your config `w_bit` will replaced to 4 instead.")
        if config.a_bit != 8:
            config.a_bit = 8
            msmodelslim_logger.warning("Running in sparse requires `a_bit` of 8, "
                                       "your config `a_bit` will replaced to 8 instead.")
    else:
        check_number(config.pr, float, 0, 1, param_name="pr")
        if config.w_bit not in W_BIT_LIST:
            raise ValueError(f"w_bit must be among choice {W_BIT_LIST}, please check it.")
        if config.a_bit not in A_BIT_LIST:
            raise ValueError(f"a_bit must be among choice {A_BIT_LIST}, please check it.")


def check_lowbit_config(config):
    check_type(config.do_smooth, bool, param_name="do_smooth")
    check_type(config.use_sigma, bool, param_name="use_sigma")
    check_number(config.sigma_factor, float, 3.0, 4.0, param_name="sigma factor")
    if config.act_method == 3 and config.is_lowbit:
        raise ValueError("act_method can not be 3 when running lowbit.")


def check_and_generate_config_param(config):
    """
    所有config的校验都置于该函数，便于给所有BaseConfig类调用
    Returns:

    """
    set_quant_param(config)
    set_lowbit_param(config)
    set_fa_quant_param(config)
    config.device, config.dev_id = validate_device(config.dev_type, config.dev_id, _SUPPORTED_DEVICES)
    check_type(config.w_bit, int, param_name="w_bit")
    check_type(config.a_bit, int, param_name="a_bit")
    check_number(config.act_method, int, 1, 3, param_name="act_method")
    check_type(config.w_method, str, param_name="w_method")
    check_element_type(config.disable_names, str, list, param_name="disable_names")
    check_type(config.mm_tensor, bool, param_name="mm_tensor")
    check_type(config.w_sym, bool, param_name="w_sym")
    check_type(config.disable_last_linear, bool, param_name='disable_last_linear')
    if config.a_bit == 8 and not config.w_sym:
        raise TypeError("w_sym should be True when a_bit = 8, please check it")
    check_type(config.co_sparse, bool, param_name="co_sparse")
    check_type(config.is_lowbit, bool, param_name='is_lowbit')
    check_type(config.open_outlier, bool, param_name="open_outlier")
    check_type(config.disable_last_linear, bool, param_name='disable_last_linear')
    check_type(config.use_kvcache_quant, bool, param_name="use_kvcache_quant")
    check_type(config.group_size, int, param_name="group_size")
    check_number(config.percdamp, float, 0, 1, param_name="percdamp")

    if config.use_kvcache_quant and config.use_fa_quant:
        raise ValueError("KV-cache and FA cannot be quantized at the same time!")

    check_sparse_config(config)
    check_lowbit_config(config)
    check_dynamic_config(config)
    set_per_group_param(config)

    config.model_quant_type = QuantType.get_quant_type(w_bit=config.w_bit, a_bit=config.a_bit,
                                                       is_sparse=config.co_sparse,
                                                       is_dynamic=config.is_dynamic,
                                                       is_lowbit=config.is_lowbit)
    WeightQuantMethod.check_quant_type(config.model_quant_type, w_method=config.w_method)
    config.w_hessian, config.hqq = WeightQuantMethod.get_wmethod_config(config.w_method)