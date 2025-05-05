import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_pos_ch=3, input_views_ch=2, skip=4, output_ch=4):
        super(NeRF, self).__init__()
        self.pos_ch = input_pos_ch
        self.views_ch = input_views_ch
        self.skip = skip

        self.main = nn.ModuleList()
        self.main.append(nn.Linear(input_pos_ch, W))
        for i in range(D-1):
            if i == skip:
                self.main.append(nn.Linear(W + input_pos_ch, W))
            else:
                self.main.append(nn.Linear(W, W))

        self.feature = nn.Linear(W, W)
        self.sigma = nn.Sequential(nn.Linear(W, 1), nn.ReLU())
        self.rgb = nn.Sequential(
            nn.Linear(W + input_views_ch, W//2),
            nn.ReLU(),
            nn.Linear(W//2, 3)
        )

    def forward(self, x, d):
        pos, views = x, d
        h = pos
        for i, l in enumerate(self.main):
            h = self.main[i](h)
            h = F.relu(h)
            if i == self.skip:
                h = torch.cat([pos, h], -1)
        sigma = self.sigma(h)
        h = self.feature(h)
        h = torch.cat([h, views], -1)
        rgb = self.rgb(h)
        return torch.cat([rgb, sigma], -1)

    
'''
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_pos_ch=3, input_views_ch=2, output_ch=4):
        super(NeRF, self).__init__()
        self.pos_ch = input_pos_ch
        self.views_ch = input_views_ch
        
        # 主干网络
        layers = [nn.Linear(input_pos_ch, W)]
        for i in range(D-1):
            layers += [nn.ReLU(), nn.Linear(W, W)]
        self.main = nn.Sequential(*layers)
        
        # 跳跃连接
        self.skip = nn.Linear(input_pos_ch + W, W)
        
        # 输出颜色和密度
        self.sigma = nn.Sequential(nn.Linear(W, 1), nn.ReLU())
        self.rgb = nn.Sequential(
            nn.Linear(W + input_views_ch, W//2),
            nn.ReLU(),
            nn.Linear(W//2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, d):
        input_pts, input_views = x, d
        h = self.main(input_pts)
        h = torch.cat([input_pts, h], -1)
        h = self.skip(h)
        sigma = self.sigma(h)
        h = torch.cat([h, input_views], -1)
        rgb = self.rgb(h)
        return torch.cat([rgb, sigma], -1)

'''