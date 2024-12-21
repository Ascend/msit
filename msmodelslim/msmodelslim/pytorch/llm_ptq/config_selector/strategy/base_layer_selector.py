from abc import ABC

from msmodelslim.pytorch.llm_ptq.config_selector.config_selector import LayerConfigSelector
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import WeightQuantMethod, QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class W8A8SStaticKMeansSelector(LayerConfigSelector, ABC):

    @staticmethod
    def mix_type() -> str:
        return 'w8a8s_static_kmeans'

    def select(self, *args, **kwargs):
        layer_name = kwargs.pop('layer_name')
        config = kwargs['layer_configs'][layer_name]
        config.w_bit = 8
        config.a_bit = 8
        config.is_dynamic = False
        config.model_quant_type = QuantType.W8A8S
        config.w_method = WeightQuantMethod.KMeans


class W8A8Selector(LayerConfigSelector, ABC):

    @staticmethod
    def mix_type() -> str:
        return 'w8a8'

    def select(self, *args, **kwargs):
        layer_name = kwargs.pop('layer_name')
        config = kwargs['layer_configs'][layer_name]
        config.w_bit = 8
        config.a_bit = 8
        config.is_dynamic = False
        config.model_quant_type = QuantType.W8A8
        config.w_method = config.w_method if config.w_method != WeightQuantMethod.KMeans else WeightQuantMethod.MinMax


class FloatSelector(LayerConfigSelector, ABC):

    @staticmethod
    def mix_type() -> str:
        return 'float'

    def select(self, *args, **kwargs):
        layer_name = kwargs.pop('layer_name')
        config = kwargs['layer_configs'][layer_name]
        config.w_bit = 16
        config.a_bit = 16
        config.model_quant_type = QuantType.FLOAT


class W4A8Selector(LayerConfigSelector, ABC):

    @staticmethod
    def mix_type() -> str:
        return 'w4a8'

    def select(self, *args, **kwargs):
        layer_name = kwargs.pop('layer_name')
        config = kwargs['layer_configs'][layer_name]
        config.w_bit = 4
        config.a_bit = 8
        config.is_dynamic = False
        config.model_quant_type = QuantType.W4A8
