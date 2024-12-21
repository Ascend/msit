__all__ = ['register_selector', 'select_layer_config']

from .config_selector import register_selector, select_layer_config
from .strategy.w8a8s_static_kmeans_w8a8_float import W8A8SStaticKMeansW8A8Float
from .strategy.w4a8_w8a8_float import W4A8W8A8Float

register_selector(W8A8SStaticKMeansW8A8Float())
register_selector(W4A8W8A8Float())
