import torch
import platform

def get_device():
    # 优先检测 CUDA（Linux 或 Windows）
    if torch.cuda.is_available():
        return "auto", torch.device("cuda")
    # macOS 专用 MPS 检测（仅当系统为 macOS 且 MPS 可用时）
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        # return torch.device("mps")
        # mps not available
        # return "cpu", torch.device("cpu")
        # mps available
        return "mps", torch.device("mps")
    # 默认回退到 CPU
    else:
        return torch.device("cpu")

if __name__=="__main__":
    device_map, device = get_device()
    print(f"当前计算设备: {device_map, device}")
