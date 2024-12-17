import torch.nn
from transformers import DynamicCache

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.kmeans.kvcache_smooth import smooth_key_cache

USE_CACHE = 'use_cache'
KEY_CACHE_MAX = 'key_cache_max'
SMOOTH_ALPHA = 'smooth_alpha'
DEFAULT_SMOOTH_ALPHA = 1.0


def use_kvcache(_, args: tuple, kwargs: dict) -> (tuple, dict):
    """
    turn on use_kvcache in pytorch prehook
    """
    if USE_CACHE not in kwargs:
        raise ValueError("module input has no use_cache, please check whether module is attention")
    kwargs[USE_CACHE] = True
    return args, kwargs


def catch_key_cache_max(module: torch.nn.Module, _: tuple, output, name, act_stats) -> None:
    """
    catch max of key cache in pytorch hook
    """
    kv_cache = output[-1]
    if not isinstance(kv_cache, DynamicCache):
        raise TypeError('past_key_value only accept DynamicCache for now, please check transformers version')
    if not hasattr(module, 'layer_idx'):
        raise ValueError('module has no layer_idx, please check whether module is attention')
    layer_kv_cache = kv_cache[module.layer_idx]
    key_cache = layer_kv_cache[0]

    # get max of key cache, key shape: (bs, head num, seq len, head dim)
    key_scale = key_cache.abs().max(dim=0)[0].max(dim=1)[0]
    key_scale = key_scale.view(-1)

    if name not in act_stats:
        act_stats[name] = {}
    module_state = act_stats[name]

    if KEY_CACHE_MAX not in module_state:
        module_state[KEY_CACHE_MAX] = key_scale
    else:
        module_state[KEY_CACHE_MAX] = torch.max(module_state[KEY_CACHE_MAX], key_scale)


def smooth_kv_cache(cfg, model: torch.nn.Module, act_stats, attention_class, logger=None):
    logger = logger or msmodelslim_logger

    smooth_alpha = getattr(cfg, SMOOTH_ALPHA, DEFAULT_SMOOTH_ALPHA)
    logger.info(f'SmoothQuant: smooth_kv_cache with alpha={smooth_alpha}')

    for name, module in model.named_modules():
        if not isinstance(module, attention_class):
            continue

        if name not in act_stats:
            logger.warning(f"attention {name} skip out of lack of act stats")
            continue

        num_key_value_groups = module.num_key_value_groups
        head_dim = module.head_dim
        num_key_value_heads = module.num_key_value_heads

        q_proj = module.q_proj
        k_proj = module.k_proj
        stats = act_stats[name]

        smooth_key_cache(k_proj, q_proj, stats, head_dim, num_key_value_groups, num_key_value_heads, smooth_alpha)
