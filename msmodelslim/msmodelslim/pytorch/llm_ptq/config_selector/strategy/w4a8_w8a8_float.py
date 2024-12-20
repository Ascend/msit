from msmodelslim.pytorch.llm_ptq.config_selector.strategy.base_layer_selector import W4A8Selector, W8A8Selector, \
    FloatSelector
from msmodelslim.pytorch.llm_ptq.config_selector.strategy.mix_layer_param_range_selector import \
    MixLayerParamRangeSelector


class W4A8W8A8Float(MixLayerParamRangeSelector):
    """
    W8A8S-Static-KMeans/W8A8/Float三类型层间混精选择。
    """

    def target_mix_type_set(self) -> set:
        return {W4A8Selector.mix_type(), W8A8Selector.mix_type(), FloatSelector.mix_type()}

    def setup(self, *args, **kwargs):
        """
        通过两个阈值进行W8-Lut/W8/Float三种类型的层间混精选择。
        -∞ <= layer_param <= threshold1:            QuantType.W8A8S with KMeans
        threshold1 <= layer_param <= threshold2:    QuantType.W8A8
        threshold2 <= layer_param <= +∞:            QuantType.FLOAT
        """

        threshold1 = kwargs.pop('threshold1')
        use_alpha = kwargs.pop('use_alpha') if 'use_alpha' in kwargs else False
        threshold2 = threshold1 * kwargs.pop('alpha') if use_alpha else kwargs.pop('threshold2')

        self.add_range_selector(min_value=float('-inf'), max_value=threshold1, selector=W4A8Selector())
        self.add_range_selector(min_value=threshold1, max_value=threshold2, selector=W8A8Selector())
        self.add_range_selector(min_value=threshold2, max_value=float('inf'), selector=FloatSelector())
