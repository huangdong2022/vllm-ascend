import torch
from torch.distributed import ProcessGroup

_HCOMM_INFO = None


def get_hcomm_info(group: ProcessGroup) -> str:
    """Get the HCCL communication information for the given group."""
    global _HCOMM_INFO
    if _HCOMM_INFO is not None:
        return _HCOMM_INFO

    rank = torch.distributed.get_rank(group)
    if torch.__version__ > "2.0":
        global_rank = torch.distributed.get_global_rank(group, rank)
        _HCOMM_INFO = group._get_backend(
            torch.device("npu")).get_hccl_comm_name(global_rank)
    else:
        _HCOMM_INFO = group.get_hccl_comm_name(rank)
    return _HCOMM_INFO
