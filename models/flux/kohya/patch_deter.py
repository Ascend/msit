from functools import wraps
import torch

def cpu_rand_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        orig_device = kwargs['device']
        if orig_device is None:
            orig_device = 'cpu'
        kwargs['device'] = 'cpu'
        res = func(*args, **kwargs)
        res = res.to(orig_device)
        return res
    
    return wrapper


def cpu_like_wrapper(func):
    @wraps(func)
    def wrapper(target_tensor, *args, **kwargs):
        orig_device = target_tensor.device
        res = func(target_tensor.to('cpu'), *args, **kwargs)
        res = res.to(orig_device)
        return res
    
    return wrapper


torch.rand = cpu_rand_wrapper(torch.rand)
torch.randn_like = cpu_like_wrapper(torch.randn_like)