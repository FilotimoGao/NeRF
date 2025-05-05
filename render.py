import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from nerf import NeRF
from nerf import get_rays
from nerf import render_rays
from data import load_data
from data import load_eval_data

# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def render_image(model, H, W, focal, pose, batch_size=6400):
    # 将 pose 转换为 GPU 张量（如果尚未转换）
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)
    
    # 计算光线原点和方向
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
    print(f"rays_d = {rays_d}")
    print(f"rays_o = {rays_o}")
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # 逐批处理光线
    rgb_map = []
    batch_size = min(batch_size, rays_o.shape[0])  # 确保不超过总光线数
    for start in range(0, rays_o.shape[0], batch_size):
        print(start)
        end = min(start + batch_size, rays_o.shape[0])
        rgb_batch = render_rays(model, rays_o[start:end], rays_d[start:end], near=3, far=5)
        rgb_map.append(rgb_batch)
    
    # 拼接结果并转换为图像
    rgb_map = torch.cat(rgb_map, dim=0)
    print(rgb_map)
    img = (rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
    
    # 清理显存
    # torch.cuda.empty_cache()
    
    return img

def remove_module_prefix(state_dict):
    """移除 state_dict 中的 'module.' 前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # 移除 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def load_model(model, optimizer, checkpoint_path):
    """加载模型权重和优化器状态"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # 修正 state_dict 的键名
        checkpoint['model_state_dict'] = remove_module_prefix(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
        return epoch
    else:
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return 0  # 如果检查点不存在，从头开始训练


if __name__ == "__main__":
    # 重新实例化模型，并不使用 GPU
    model = NeRF(input_pos_ch=63, input_views_ch=27).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # 加载模型权重，并将其映射到 CPU
    model_path = "saved_models/checkpoint_iter_15000.pth"
    epoch = load_model(model, optimizer, model_path)
    model.eval()  # 设置模型为评估模式

    # 加载数据以获取图像、姿态和其他参数
    datadir = "nerf_recon_dataset/nerf_synthetic/hotdog"
    image, pose, H, W, focal = load_eval_data(datadir, index = 19)

    # 渲染图像
    img = render_image(model, H, W, focal, pose)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'render_result.png')
    plt.show()
