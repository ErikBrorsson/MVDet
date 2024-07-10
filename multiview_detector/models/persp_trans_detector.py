import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        print("# cameras in model: ", self.num_cam)
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass


    def forward(self, imgs, proj_mats, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for i in range(N):
            img_feature = self.base_pt1(imgs[:, i].to('cuda:0'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)
            proj_mat = proj_mats[i].repeat([B, 1, 1]).float().to('cuda:0')

            # here, the proj_mat has been constructed for a specific grid (output) and image (input) size.
            # it is critical that the shape of img_feature equals the intended input size, and that self.reducedgrid_shape specifies the intended output size.
            world_feature = kornia.geometry.transform.warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape) # reducedgrid_shape=[480/4, 1440/4]
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.show()
        map_result = self.map_classifier(world_features.to('cuda:0'))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return map_result, imgs_result

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
