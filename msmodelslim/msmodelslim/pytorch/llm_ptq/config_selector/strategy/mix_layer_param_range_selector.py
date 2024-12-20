import abc
from abc import ABC
from collections import namedtuple
from typing import List, Tuple, Dict

from ascend_utils.common.security import check_element_type
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.config_selector.config_selector import LayerConfigDictSelector
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig

Range = Tuple[float, float]
LayerParam = Tuple[float, str]
LayerConfigDict = Dict[str, QuantConfig]

LayerContex = namedtuple('LayerContex', ['config', 'param'])
GlobalContex = namedtuple('GlobalContex', ['layers', 'threshold1', 'threshold2'])
RangeSelector = namedtuple('RangeSelector', ['min_value', 'max_value', 'selector'])


class MixLayerParamRangeSelector(LayerConfigDictSelector, ABC):

    def __init__(self):
        super().__init__()
        self.config_selectors: List[RangeSelector] = []

    @abc.abstractmethod
    def target_mix_type_set(self) -> set:
        pass

    def add_range_selector(self, min_value, max_value, selector):
        self.config_selectors.append(RangeSelector(min_value=min_value, max_value=max_value, selector=selector))

    def select(self, *args, **kwargs):
        layer_configs = kwargs['layer_configs']
        layer_params = kwargs['layer_params']
        for layer_param, name in layer_params:
            if name not in layer_configs:
                msmodelslim_logger.warning(f'skip {name} because it is not in layer config map')
                continue
            for range_selector in self.config_selectors:
                if range_selector.min_value <= layer_param <= range_selector.max_value:
                    msmodelslim_logger.debug(
                        f'{name} will use selector {type(range_selector.selector)} because {range_selector.min_value} <= {layer_param} <= {range_selector.max_value}')
                    range_selector.selector.select(layer_name=name, **kwargs)

    def match(self, *args, **kwargs) -> bool:

        if 'mix_method' not in kwargs:
            msmodelslim_logger.debug(f'Not match because mix_method is not specified')
            return False

        if kwargs['mix_method'] != 'auto':
            msmodelslim_logger.debug(f'Not match because mix_method is not "auto"')
            return False

        if 'mix_types' not in kwargs:
            msmodelslim_logger.debug(f'Not match because mix_types is not specified')
            return False

        mix_types = kwargs.pop('mix_types')

        try:
            check_element_type(mix_types, str, list)
        except (ValueError, TypeError) as e:
            msmodelslim_logger.debug(f'Not match because {e}')
            return False

        if set(mix_types) != self.target_mix_type_set():
            msmodelslim_logger.debug(f'Not match because {set(mix_types)} != {self.target_mix_type_set()}')
            return False

        msmodelslim_logger.info(f'Layer selector match {type(self)}')

        return True
