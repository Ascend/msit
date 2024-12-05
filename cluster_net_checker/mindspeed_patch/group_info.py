import torch
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

import megatron.core.parallel_state as ps
from megatron.training import get_args


LOCAL_RANK_MAP = {
    "pp": ps.get_pipeline_model_parallel_rank,
    "tp": ps.get_tensor_model_parallel_rank,
    "ep": ps.get_expert_model_parallel_rank,
    "dp": ps.get_data_parallel_rank
}


def get_golbal_rank(group):
    return torch.distributed.get_global_rank(group, torch.distributed.get_rank(group))


class GroupInfo:
    def __init__(self) -> None:
        self.tp_group = ps.get_tensor_model_parallel_group()
        self.pp_group = ps.get_pipeline_model_parallel_group()
        self.ep_group = ps.get_expert_model_parallel_group()
        self.dp_group = ps.get_data_parallel_group()

        tp_rank = get_golbal_rank(self.tp_group)
        pp_rank = get_golbal_rank(self.pp_group)
        assert pp_rank == tp_rank
        self.global_rank = tp_rank
        self.group_dict = {
            "pp": self.pp_group,
            "tp": self.tp_group,
            "ep": self.ep_group,
            "dp": self.dp_group
        }
        self.group_map_list = self.get_group_map()
        self.rank_map_list = self.get_rank_map()
        self.args_list = self.get_args()

    def get_group_map(self):
        res_list = []
        for key, value in self.group_dict.items():
            group_name = str(value._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank))
            res_list.append(
                str({group_name: key})
            )
        return res_list

    def get_rank_map(self):
        res_list = []
        for key, value in self.group_dict.items():
            local_rank = LOCAL_RANK_MAP[key]()
            group_name = str(value._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank))
            res_list.append(
                str({group_name: str({local_rank: self.global_rank})})
            )
        return res_list
    
    def get_args(self):
        args = vars(get_args())

        dump_args = [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "data_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "sequence_parallel",
            "rank",
            "world_size"
        ]

        res_list = []
        for key in dump_args:
            res_list.append(str({key:args.get(key, "None")}))
        return res_list

    def get_group_info(self):
        res_list = []
        for item in self.group_map_list:
            res_list.append(f"communication_group_name_map: {item}")
        for item in self.rank_map_list:
            res_list.append(f"communication_group_rank_map: {item}")
        for item in self.args_list:
            res_list.append(f"distributed_args: {item}")
        return res_list



"""
class GroupInfo:
    def __init__(self) -> None:
        self.tp_group = ps.get_tensor_model_parallel_group()
        self.pp_group = ps.get_pipeline_model_parallel_group()
        self.ep_group = ps.get_expert_model_parallel_group()
        self.dp_group = ps.get_data_parallel_group()

        tp_rank = get_golbal_rank(self.tp_group)
        pp_rank = get_golbal_rank(self.pp_group)
        assert pp_rank == tp_rank
        self.global_rank = tp_rank
        self.group_dict = {
            "pp": self.pp_group,
            "tp": self.tp_group,
            "ep": self.ep_group,
            "dp": self.dp_group
        }

    def get_group_map(self):
        res_list = []
        for key, value in self.group_dict.items():
            group_name = str(value._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank))
            res_list.append(
                str({key: group_name})
            )

        res_str = ", ".join(res_list)

        
        return res_str

    def get_rank_map(self):
        res_list = []
        for key, value in self.group_dict.items():
            local_rank = LOCAL_RANK_MAP[key]()
            group_name = str(value._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank))
            res_list.append(
                str({group_name: str({local_rank: self.global_rank})})
            )
        res_str = ", ".join(res_list)
        return res_str
    
    def get_args(self):
        args = vars(get_args())

        dump_args = [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "data_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "sequence_parallel",
            "rank",
            "world_size"
        ]

        res_dict = {}
        for key in dump_args:
            res_dict[key] = args.get(key, "None")
        return str(res_dict)
"""
