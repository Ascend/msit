from .pruning.importance import MagnitudeImportance, TaylorImportance
from .pruning.ascend_pruner import AscendPruner

from .pruning.smcp_pruner import SMCPPruner
from .pruning.base_pruner import BasePruner
from .pruning.customized_pruners import SwinSelfAttentionPruner, SwinWindowAttentionPruner, SwinPatchMergingPruner
from .td.td import apply_td, solve_prune_td

__all__ = ["MagnitudeImportance", "TaylorImportance", "AscendPruner", "SMCPPruner",
           "SwinSelfAttentionPruner", "SwinWindowAttentionPruner", "SwinPatchMergingPruner"]