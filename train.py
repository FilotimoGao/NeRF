import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from nerf import NeRF
from nerf import get_rays
from nerf import render_rays
from data import load_data
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 封装模型到 DataParallel
model = NeRF(input_pos_ch=63, input_views_ch=27)
# model = NeRF()
if device != "cpu":
    model = nn.DataParallel(model)
model.to(device)

os.makedirs("render", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
N_iter = 40000  # 迭代次数
batch_size = 4096  # 每批光线数

@torch.no_grad()
def render_image(model, H, W, focal, pose, batch_size=6400):
    # 将 pose 转换为 GPU 张量（如果尚未转换）
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose).cuda()
    
    # 计算光线原点和方向
    rays_o, rays_d = get_rays(H, W, focal, pose)
    # print(f"rays_d = {rays_d}")
    # print(f"rays_o = {rays_o}")
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # 逐批处理光线
    rgb_map = []
    batch_size = min(batch_size, rays_o.shape[0])  # 确保不超过总光线数
    for start in range(0, rays_o.shape[0], batch_size):
        end = min(start + batch_size, rays_o.shape[0])
        rgb_batch = render_rays(model, rays_o[start:end], rays_d[start:end])
        rgb_map.append(rgb_batch)
    
    # 拼接结果并转换为图像
    rgb_map = torch.cat(rgb_map, dim=0)
    # print(f"rgb_map = {rgb_map}")
    img = (rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
    # img = rgb_map.reshape(H, W, 3).cpu().numpy()
    
    # 清理显存
    torch.cuda.empty_cache()
    
    return img


def train(resume=False, checkpoint_path=None):
    datadir = "nerf_recon_dataset/nerf_synthetic/hotdog"
    images, poses, H, W, focal = load_data(datadir)

    # 如果选择从检查点恢复训练
    if resume and checkpoint_path is not None:
        start_iter = load_model(model, optimizer, checkpoint_path)  # 加载模型和优化器状态
    else:
        start_iter = 0  # 如果不恢复训练，则从 0 开始

    for i in tqdm(range(start_iter, N_iter), desc="Training Progress"):
        img_idx = np.random.randint(len(images))
        target = torch.tensor(images[img_idx]).to(device)

        rays_o, rays_d = get_rays(H, W, focal, torch.tensor(poses[img_idx]).to(device))
        
        rays_idx = torch.randperm(H*W)[:batch_size]
        rays_o_batch = rays_o.reshape(-1, 3)[rays_idx]
        rays_d_batch = rays_d.reshape(-1, 3)[rays_idx]
        target_batch = target.reshape(-1, 3)[rays_idx]
        
        rgb_map = render_rays(model, rays_o_batch, rays_d_batch)
        loss = ((rgb_map - target_batch) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # 每次优化后清理显存
        
        if i % 500 == 0:
            print(f"Iter {i}, Loss: {loss.item():.4f}")
            test_pose = poses[0]
            img = render_image(model, H, W, focal, test_pose)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(f'render/render_result_iter_{i}.png')
            plt.show()
            
            # 保存模型权重
            checkpoint = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

            torch.save(checkpoint, f"saved_models/checkpoint_iter_{i}.pth")
            print(f"Model saved at iteration {i}")
            

def load_model(model, optimizer, checkpoint_path):
    """加载模型权重和优化器状态"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
        return epoch  # 返回加载的 epoch，以便继续训练
    else:
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return 0  # 如果检查点不存在，从头开始训练


if __name__=="__main__":
    resume_training = True  # 设为 True 表示要恢复训练
    checkpoint_path = "saved_models/checkpoint_iter_19500.pth"  # 替换为你的检查点路径

    train(resume=resume_training, checkpoint_path=checkpoint_path)