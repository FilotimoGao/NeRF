import torch
import numpy as np
import os
from nerf import NeRF
from .pos_decode import position_decoding

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_L = 10
dir_L = 4

def get_rays(H, W , focal, pose):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H),
        indexing = 'xy'
    )
    i, j = i.to(device), j.to(device)
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    rays_o = pose[:3, -1].expand_as(rays_d)
    return rays_o, rays_d

def render_rays(model, rays_o, rays_d, near=2, far=6, samples_num=64):
    depth = near + (far - near) * torch.rand(rays_o.shape[:-1] + (samples_num,)).to(device)

    pos = rays_o[..., None, :] + rays_d[..., None, :] * depth[..., :, None]
    pos_flat = pos.reshape(-1, 3)
    dir = rays_d[:, None].expand(pos.shape)
    dir_flat = dir.reshape(-1, 3)

    pos_decoded = position_decoding(pos_flat, pos_L).float()
    dir_decoded = position_decoding(dir_flat, dir_L).float()

    # print(f"pos_decoded = {pos_decoded}")
    # print(f"dir_decoded = {dir_decoded}")

    outputs = model(pos_decoded, dir_decoded)
    outputs = outputs.reshape(*pos.shape[:-1], 4) # 推测为W*H*64*4
    rgb, sigma = outputs[..., :3],  outputs[..., 3]

    # 体渲染
    deltas = depth[..., 1:] - depth[..., :-1]
    deltas_inf = 1e10 * torch.ones_like(deltas[..., :1])
    deltas = torch.cat([deltas, deltas_inf], -1)
    alphas = 1. - torch.exp(-sigma * deltas)
    weights = alphas * torch.cumprod(torch.cat([torch.ones_like(alphas[..., :1]), 1. - alphas + 1e-10], -1), -1)[..., :-1]
    # print(f"weights.shape = {weights.shape}")
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map

if __name__=="__main__":
    H, W = 4, 4
    focal =  0.5 * W / np.tan(0.5 * 0.6911112070083618)
    pose = [
                [
                    -0.9938939213752747,
                    -0.10829982906579971,
                    0.021122142672538757,
                    0.08514608442783356
                ],
                [
                    0.11034037917852402,
                    -0.9755136370658875,
                    0.19025827944278717,
                    0.7669557332992554
                ],
                [
                    0.0,
                    0.19142703711986542,
                    0.9815067052841187,
                    3.956580400466919
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    rays_o, rays_d = get_rays(H, W, focal, torch.tensor(pose))
    render_rays(NeRF(input_pos_ch = 63, input_views_ch = 27), rays_o, rays_d)