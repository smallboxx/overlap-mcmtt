import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn.functional as F
import cv2
from PIL import Image

def array2heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return heatmap

def overlab_MaskList(world_features, B, S, vis=False):
    t_ = []
    for cam in range(S):
        t_world  = torch.norm(world_features[cam * B].detach(), dim=0).cpu()
        t_.append(np.where(t_world > 0, 1, 0))
    overlap = t_[0]
    for indx in range(S-1):
        overlap += t_[indx+1]

    if S==6:
        mask1 = np.where(overlap==1, 1, 0) + np.where(overlap==2, 1, 0) + np.where(overlap==3, 1, 0)
        mask2 = np.where(overlap==4, 1, 0) + np.where(overlap==5, 1, 0)
        mask3 = np.where(overlap==6, 1, 0)
        MaskList = [mask1, mask2, mask3]
    else:
        mask1 = np.where(overlap==1, 1, 0) + np.where(overlap==2, 1, 0) + np.where(overlap==3, 1, 0)
        mask2 = np.where(overlap==4, 1, 0) + np.where(overlap==5, 1, 0)
        mask3 = np.where(overlap==6, 1, 0) + np.where(overlap==7, 1, 0)
        MaskList = [mask1, mask2, mask3]
    if vis:
        for mask in MaskList:
            m_img  = array2heatmap(mask)
            plt.imshow(m_img)
            plt.show()

    return MaskList

def create_coord_map(img_size, with_r=False):
    H, W = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        grid_r = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, grid_r], dim=1)
    return ret

class ConvWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, stride=2, reduction=None):
        super(ConvWorldFeat, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, stride, 1), nn.ReLU(), )
        self.coord_map = create_coord_map(np.array(Rworld_shape) // stride)
        self.reduction = reduction
        if self.reduction is None:
            combined_input_dim = base_dim * num_cam + 2
        elif self.reduction == 'sum':
            combined_input_dim = base_dim + 2
        else:
            raise Exception
        self.world_feat = nn.Sequential(nn.Conv2d(combined_input_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(), )
        self.upsample = nn.Sequential(nn.Upsample(Rworld_shape, mode='bilinear', align_corners=False),
                                      nn.Conv2d(hidden_dim, base_dim, 3, 1, 1), nn.ReLU(), )

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        x = self.downsample(x.view(B * N, C, H, W))
        _, _, H, W = x.shape
        if self.reduction is None:
            x = x.view(B, N * C, H, W)
        elif self.reduction == 'sum':
            x = x.sum(dim=1)
        else:
            raise Exception
        x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1)
        x = self.world_feat(x)
        x = self.upsample(x)
        return x


class OverLapDecoder(nn.ModuleList):
    def __init__(self, in_channels, n_classes, n_ids, Y, X, num_cameras):
        super().__init__()
        self.reid_feat = 64
        self.base_dim  = 128
        self.feat2d = 128
        shared_out_channels = in_channels
        self.overlap_low = ConvWorldFeat(num_cameras, (Y,X), self.base_dim)
        self.overlap_mid = ConvWorldFeat(num_cameras, (Y,X), self.base_dim)
        self.overlap_high= ConvWorldFeat(num_cameras, (Y,X), self.base_dim)
        # bev
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(self.base_dim, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(self.base_dim, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
        )
        self.instance_center_head[-1].bias.data.fill_(-2.19)

        self.instance_size_head = nn.Sequential(
            nn.Conv2d(self.base_dim, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 3, kernel_size=1, padding=0),
        )
        self.instance_rot_head = nn.Sequential(
            nn.Conv2d(self.base_dim, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 8, kernel_size=1, padding=0),
        )

        # img
        self.img_center_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, n_classes, kernel_size=1, padding=0),
        )
        self.img_offset_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )
        self.img_size_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )

        # re_id
        self.id_feat_head = nn.Sequential(
            nn.Conv2d(self.feat2d, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.reid_feat, kernel_size=1, padding=0),
        )
        self.img_id_feat_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, self.reid_feat, kernel_size=1, padding=0),
        )
        self.emb_scale = math.sqrt(2) * math.log(n_ids - 1)

    def forward(self, w_feats, MaskList, feat_cams):
        b, s, c, h, w = w_feats.shape
        mask1 = torch.from_numpy(MaskList[0]).to(w_feats.device)
        mask2 = torch.from_numpy(MaskList[1]).to(w_feats.device)
        mask3 = torch.from_numpy(MaskList[2]).to(w_feats.device)

        m_feat_low = self.overlap_low(w_feats * mask1)
        m_feat_mid = self.overlap_mid(w_feats * mask2)
        m_feat_high= self.overlap_high(w_feats * mask3)

        out_x = m_feat_low+m_feat_mid+m_feat_high

        instance_center_output = self.instance_center_head(out_x)
        instance_offset_output = self.instance_offset_head(out_x)
        instance_size_output = self.instance_size_head(out_x)
        instance_rot_output = self.instance_rot_head(out_x)
        instance_id_feat_output = self.emb_scale * F.normalize(self.id_feat_head(out_x), dim=1)

        img_center_output = self.img_center_head(feat_cams)  # B*S,1,H/8,W/8
        img_offset_output = self.img_offset_head(feat_cams)  # B*S,2,H/8,W/8
        img_size_output = self.img_size_head(feat_cams)  # B*S,2,H/8,W/8
        img_id_feat_output = self.emb_scale * F.normalize(self.img_id_feat_head(feat_cams), dim=1)  # B*S,C,H/8,W/8
        return {
            # bev
            'raw_feat': out_x,
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_size': instance_size_output.view(b, *instance_size_output.shape[1:]),
            'instance_rot': instance_rot_output.view(b, *instance_rot_output.shape[1:]),
            'instance_id_feat': instance_id_feat_output.view(b, *instance_id_feat_output.shape[1:]),
            # img
            'img_center': img_center_output,
            'img_offset': img_offset_output,
            'img_size': img_size_output,
            'img_id_feat': img_id_feat_output,
        }