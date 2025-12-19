import gc
import torch

if torch.cuda.is_available():
    # clear VRAM   
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()