from typing import Sequence
import torch.nn as nn

from transformers.models.swin.modeling_swin import SwinSelfAttention
from timm.models.swin_transformer import WindowAttention
from torch_pruning import function
from transformers.pytorch_utils import Conv1D


class SwinSelfAttentionPruner(function.LinearPruner):
    TARGET_MODULES = SwinSelfAttention

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        if (layer.all_head_size - len(idxs)) % layer.num_attention_heads == 0:
            raise Exception(
                "all_head_size (%d) of SwinSelfAttention after pruning must divide evenly by `attention_head_size` (%d)" % (
                layer.all_head_size, layer.num_attention_heads))

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.all_head_size)) - set(idxs))
        keep_idxs.sort()

        layer.query = super().prune_out_channels(layer.query, idxs)
        layer.key = super().prune_out_channels(layer.key, idxs)
        layer.value = super().prune_out_channels(layer.value, idxs)

        layer.num_attention_heads = layer.query.out_features // layer.attention_head_size
        layer.all_head_size = layer.query.out_features

        tile_relative_position_bias_table = layer.relative_position_bias_table.tile(1, layer.attention_head_size)
        pruned_parameter = self._prune_parameter_and_grad(tile_relative_position_bias_table,
                                                          keep_idxs,
                                                          1)

        bias_table_idxs = [i * layer.attention_head_size for i in range(layer.num_attention_heads)]
        final_pruned_tensor = pruned_parameter.data[:, bias_table_idxs]

        layer.relative_position_bias_table = nn.Parameter(final_pruned_tensor)
        return layer

    def prune_in_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.all_head_size)) - set(idxs))
        keep_idxs.sort()

        layer.query = super().prune_in_channels(layer.query, idxs)
        layer.key = super().prune_in_channels(layer.key, idxs)
        layer.value = super().prune_in_channels(layer.value, idxs)

    def get_out_channels(self, layer):
        return layer.all_head_size

    def get_in_channels(self, layer):
        return layer.query.in_features


class SwinWindowAttentionPruner(function.LinearPruner):
    TARGET_MODULES = WindowAttention

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        if (layer.dim - len(idxs)) % layer.num_heads == 0:
            raise Exception("dim (%d) of WindowAttention after pruning must divide evenly by `num_heads` (%d)" % (
            layer.dim, layer.num_heads))

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.dim)) - set(idxs))
        keep_idxs.sort()

        qkv_idxs = []
        for i in range(3):
            for idx in keep_idxs:
                qkv_idxs.append(idx + i * layer.dim)

        head_dim = layer.dim // layer.num_heads
        layer.qkv = super().prune_out_channels(layer.qkv, qkv_idxs)
        layer.num_heads = layer.qkv.out_features // (3 * head_dim)

        layer.head_dim = layer.dim // layer.num_heads

        layer.proj = super().prune_in_channels(layer.proj, idxs)
        layer.proj = super().prune_out_channels(layer.proj, idxs)

        tile_relative_position_bias_table = layer.relative_position_bias_table.tile(1, head_dim)
        pruned_parameter = self._prune_parameter_and_grad(tile_relative_position_bias_table,
                                                          keep_idxs,
                                                          1)

        bias_table_idxs = [i * head_dim for i in range(layer.num_heads)]
        final_pruned_tensor = pruned_parameter.data[:, bias_table_idxs]

        layer.relative_position_bias_table = nn.Parameter(final_pruned_tensor)
        return layer

    def prune_in_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.dim)) - set(idxs))
        keep_idxs.sort()

        layer.qkv = super().prune_in_channels(layer.qkv, idxs)

    def get_out_channels(self, layer):
        return layer.proj.out_features  # layer.all_head_size

    def get_in_channels(self, layer):
        return layer.qkv.in_features  # layer.qkv.in_features

    def get_out_channel_groups(self, layer):
        return layer.num_heads


class SwinPatchMergingPruner(function.BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        function.prune_linear_out_channels(layer.reduction, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        dim = layer.dim
        idxs_repeated = idxs + \
                        [i + dim for i in idxs] + \
                        [i + 2 * dim for i in idxs] + \
                        [i + 3 * dim for i in idxs]
        function.prune_linear_in_channels(layer.reduction, idxs_repeated)
        function.prune_layernorm_out_channels(layer.norm, idxs_repeated)
        return layer

    def get_out_channels(self, layer):
        return layer.reduction.out_features

    def get_in_channels(self, layer):
        return layer.dim