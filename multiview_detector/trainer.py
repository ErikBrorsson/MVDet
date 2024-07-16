import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.utils.projection import get_imagecoord_from_worldcoord, get_worldcoord_from_imagecoord,\
    get_worldcoord_from_imagecoord_w_projmat, get_worldgrid_from_worldcoord

import torchvision
from multiview_detector.augmentation.homographyaugmentation import HomographyDataAugmentation


import kornia


class Augmentation:
    def __init__(self, dropview=False, permutation=False, mvaug=False,
                 grid_reduce=4, img_reduce=4, img_shape=[1080, 1920] , worldgrid_shape = [480, 1440]  # H,W; N_row,N_col
) -> None:
        self.dropview = dropview
        self.permutation = permutation
        self.mvaug = mvaug
        self.img_reduce = img_reduce
        self.img_shape = img_shape
        self.reducedgrid_shape = list(map(lambda x: int(x / grid_reduce), worldgrid_shape))



    def dropview_augment(self, imgs, map_label, imgs_labels, proj_mats):
        # imgs.shape = (1, 4, 3, 720, 1280) = (batch_size, n_cams, RGB, height, width)

        # print("imgs.shape", imgs.shape)
        # print("map_gt.shape", map_gt.shape)
        # for img_gt in imgs_gt:
        #     print("img_gt.shape", img_gt.shape)

        if self.dropview:
            # set all pixel values of the dropped image to 0
            drop_indx = np.random.choice(np.arange(imgs.shape[1]))
            imgs[:, drop_indx, :, :, :] = 0

            # set the perspective view label for the dropped view to None
            # since don't want to provide supervision on a dropped view.
            imgs_labels[drop_indx] = None

        return imgs, map_label, imgs_labels, proj_mats
    

    def camera_permutation_augment(self, imgs, map_label, imgs_labels, proj_mats):
        imgs_shuffled = torch.zeros_like(imgs)
        imgs_labels_shuffled = [None]*len(imgs_labels)
        proj_mats_shuffled = [None]*len(proj_mats)

        permutation = np.random.permutation(imgs.shape[1])
        print("permutation: ", permutation)
        # permutation = [3,0,2,1]
        for i, p in enumerate(permutation):
            imgs_shuffled[:, p, :, :] = imgs[:, i, :, :]
            imgs_labels_shuffled[p] = imgs_labels[i]
            proj_mats_shuffled[p] = proj_mats[i]

        return imgs_shuffled, map_label, imgs_labels_shuffled, proj_mats_shuffled
    

    def mvaug_augmentation(self, data, map_gt, imgs_gt, proj_mats_mvaug_features):
        
        # persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomPerspective())
        # persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(30)) # TODO set degrees
        scene_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(30)) # TODO set degrees

        # TODO there is a slight difference between map_gt_aug_temp and map_gt_aug. I'm not sure why
        # augment the map_label
        map_gt_aug_temp = scene_aug(map_gt)

        # TODO assuming batch_size 1
        map_gt_aug = torch.zeros_like(map_gt)
        foot_gt = map_gt[0, 0]
        foot_points = (foot_gt == 1).nonzero().float()
        temp = torch.zeros_like(foot_points)
        temp[:, 0] = foot_points[:, 1]
        temp[:, 1] = foot_points[:, 0]
        foot_points = temp
        foot_points_aug, pedestrian_ids = scene_aug.augment_gt_point_view_based(foot_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=map_gt.shape[-2:])
        for pos in foot_points_aug:
            map_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1

        data_aug = torch.zeros_like(data)
        proj_mat_aug_list = []
        img_gt_aug_list = []

        # proj_mat_aug_list_without_scene = []

        # loop over the images and apply augmentation
        for i, img_gt in enumerate(imgs_gt):
            persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(
                degrees = 45, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10)) # parameters set according to MVAug's proposal

            img = data[0, i, :, :]
            # proj_mat_aug = proj_mats_mvaug[i] # bev-grid reduced -> image (720x1280)

            # augment the image
            data_aug[0, i] = persp_aug(img)

            # augment the image label
            # TODO assuming batch_size 1
            img_gt_aug = torch.zeros_like(img_gt)
            foot_gt = img_gt[0, 1]
            foot_points = (foot_gt == 1).nonzero().float()
            temp = torch.zeros_like(foot_points)
            temp[:, 0] = foot_points[:, 1]
            temp[:, 1] = foot_points[:, 0]
            foot_points = temp
            foot_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(foot_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
            for pos in foot_points_aug:
                img_gt_aug[:,1,int(pos[1].item()), int(pos[0].item())] = 1
            head_gt = img_gt[0, 0]
            head_points = (head_gt == 1).nonzero().float()
            temp = torch.zeros_like(head_points)
            temp[:, 0] = head_points[:, 1]
            temp[:, 1] = head_points[:, 0]
            head_points = temp
            head_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(head_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
            for pos in head_points_aug:
                img_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1
            img_gt_aug_list.append(img_gt_aug)

            # augment the projection matrix to account for persp aug
            temp = torch.tensor([int(x / self.img_reduce) for x in self.img_shape])
            proj_mat_aug_f = persp_aug.augment_homography_view_based(proj_mats_mvaug_features[i].float(), (temp).float()) # bev-grid reduced -> warped image (720x1280)

            temp = torch.tensor(self.reducedgrid_shape)
            proj_mat_aug_f = scene_aug.augment_homography_scene_based(proj_mat_aug_f, [int(x) for x in temp])

            proj_mat_aug_list.append(torch.linalg.inv(proj_mat_aug_f))

        return data_aug, map_gt_aug, img_gt_aug_list, proj_mat_aug_list

    def strong_augmentation(self, imgs, map_label, imgs_labels, proj_mats):
        if self.permutation:
            imgs, map_label, imgs_labels, proj_mats = self.camera_permutation_augment(imgs, map_label, imgs_labels, proj_mats)
        if self.mvaug:
            imgs, map_label, imgs_labels, proj_mats = self.mvaug_augmentation(imgs, map_label, imgs_labels, proj_mats)
        if self.dropview:
            imgs, map_label, imgs_labels, proj_mats = self.dropview_augment(imgs, map_label, imgs_labels, proj_mats)
        return imgs, map_label, imgs_labels, proj_mats



class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0, augmentation_module: Augmentation=Augmentation()):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

        self.augmentation = augmentation_module


    def visualize_grid_and_bev(self, proj_mat, img, bev, f_name, foot_points, map_label=None):
        bev_h = 360
        bev_w = 120
        # x = torch.linspace(0, bev_h-1, bev_h)
        x = torch.linspace(0, bev_h-1, int(bev_h/4))
        # y = torch.linspace(0, bev_w-1, bev_w)
        y = torch.linspace(0, bev_w-1, int(bev_w/4))
        mesh = torch.meshgrid([x,y], indexing="xy")
        grid = torch.concat([mesh[0].unsqueeze(0), mesh[1].unsqueeze(0)])
        grid = grid.reshape((2, -1))
        grid_homo = torch.ones((3, grid.shape[1]))
        grid_homo[0:2, :] = grid
        grid_homo = grid_homo.unsqueeze(0)
        grid_persp = torch.bmm(proj_mat.float().to('cuda:0'), grid_homo.to('cuda:0')).cpu().numpy().squeeze()
        grid_persp = grid_persp / grid_persp[2, :]
        img = img.cpu().numpy().squeeze().transpose([1, 2, 0])
        img = Image.fromarray((img * 255).astype('uint8'))
        img = np.array(img)
        for p in grid_persp.transpose():
            img = cv2.circle(img, (int(p[0]), int(p[1])), 1, (0,0,255), -1)
        for p in foot_points:
            img = cv2.circle(img, (int(p[0]), int(p[1])), 4, (255,0,0), -1)

        fig = plt.figure(figsize=(16,9))
        subplt0 = fig.add_subplot(211, title="img")
        subplt1 = fig.add_subplot(212, title="bev_img")                
        subplt0.imshow(img)
        
        if map_label is not None:
            img0 = Image.fromarray((bev.cpu().numpy().squeeze().transpose([1, 2, 0]) * 255).astype('uint8'))
            bev_with_label = add_heatmap_to_image(map_label.cpu().detach().numpy().squeeze(), img0)
            subplt1.imshow(np.array(bev_with_label))
        else:
            subplt1.imshow(bev.cpu().numpy().squeeze().transpose([1, 2, 0]))

        plt.savefig(f_name)
        plt.close(fig)

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _, proj_mats, proj_mats_mvaug, projm_img2bevred, projm_imgred2bevred, proj_mats_mvaug_features) in enumerate(data_loader):
            optimizer.zero_grad()


            mv_aug_viz = False
            if mv_aug_viz:
                scene_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(30)) # TODO set degrees

                # # TODO remove this (just used for initialization)
                # temp = torch.tensor(data_loader.dataset.reducedgrid_shape)
                # proj_mat_aug_f = scene_aug.augment_homography_scene_based(proj_mats_mvaug_features[0].float(), [int(x) for x in temp])


                # TODO there is a slight difference between map_gt_aug_temp and map_gt_aug. I'm not sure why
                # augment the map_label
                map_gt_aug_temp = scene_aug(map_gt)

                # TODO assuming batch_size 1
                map_gt_aug = torch.zeros_like(map_gt)
                foot_gt = map_gt[0, 0]
                foot_points = (foot_gt == 1).nonzero().float()
                temp = torch.zeros_like(foot_points)
                temp[:, 0] = foot_points[:, 1]
                temp[:, 1] = foot_points[:, 0]
                foot_points = temp
                foot_points_aug, pedestrian_ids = scene_aug.augment_gt_point_view_based(foot_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=map_gt.shape[-2:])
                for pos in foot_points_aug:
                    map_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1


                # Maybe there could be some sampling that "misses" a gt point otherwise.
                # assert torch.sum(map_label_aug) == torch.sum(map_gt), "some pedestrian label vanished in scene augmentation"
                fig = plt.figure()
                subplt0 = fig.add_subplot(111, title="map_label")
                subplt0.imshow(self.criterion._traget_transform(map_gt, map_gt, data_loader.dataset.map_kernel).cpu().detach().numpy().squeeze())
                plt.savefig(os.path.join(f'imap_label.jpg'))
                plt.close(fig)

                fig = plt.figure()
                subplt1 = fig.add_subplot(111, title="map_label_aug")
                subplt1.imshow(self.criterion._traget_transform(map_gt_aug, map_gt_aug, data_loader.dataset.map_kernel).cpu().detach().numpy().squeeze())
                plt.savefig(os.path.join(f'imap_label_aug.jpg'))
                plt.close(fig)

                fig = plt.figure()
                subplt1 = fig.add_subplot(111, title="map_label_aug2")
                subplt1.imshow(self.criterion._traget_transform(map_gt_aug_temp, map_gt_aug_temp, data_loader.dataset.map_kernel).cpu().detach().numpy().squeeze())
                plt.savefig(os.path.join(f'imap_label_aug2.jpg'))
                plt.close(fig)

                # initialize augmented images, image labels, and projection matrices
                data_aug = torch.zeros_like(data)
                proj_mat_aug_list = []
                img_gt_aug_list = []

                proj_mat_aug_list_without_scene = []
                # loop over the images and apply augmentation
                for i, img_gt in enumerate(imgs_gt):
                    persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(30)) # TODO set degrees

                    img = data[0, i, :, :]
                    # proj_mat_aug = proj_mats_mvaug[i] # bev-grid reduced -> image (720x1280)

                    # augment the image
                    data_aug[0, i] = persp_aug(img)

                    # augment the image label
                    # TODO assuming batch_size 1
                    img_gt_aug = torch.zeros_like(img_gt)
                    foot_gt = img_gt[0, 1]
                    foot_points = (foot_gt == 1).nonzero().float()
                    temp = torch.zeros_like(foot_points)
                    temp[:, 0] = foot_points[:, 1]
                    temp[:, 1] = foot_points[:, 0]
                    foot_points = temp
                    foot_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(foot_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
                    for pos in foot_points_aug:
                        img_gt_aug[:,1,int(pos[1].item()), int(pos[0].item())] = 1
                    head_gt = img_gt[0, 0]
                    head_points = (head_gt == 1).nonzero().float()
                    temp = torch.zeros_like(head_points)
                    temp[:, 0] = head_points[:, 1]
                    temp[:, 1] = head_points[:, 0]
                    head_points = temp
                    head_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(head_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
                    for pos in head_points_aug:
                        img_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1
                    img_gt_aug_list.append(img_gt_aug)

                    # augment the projection matrix to account for persp aug
                    # temp = torch.tensor([data.shape[-2] // data_loader.dataset.img_reduce, data.shape[-1] // data_loader.dataset.img_reduce])
                    temp = torch.tensor([int(x / data_loader.dataset.img_reduce) for x in data_loader.dataset.img_shape])
                    proj_mat_aug_f = persp_aug.augment_homography_view_based(proj_mats_mvaug_features[i].float(), (temp).float()) # bev-grid reduced -> warped image (720x1280)
                    proj_mat_aug_list_without_scene.append(torch.linalg.inv(proj_mat_aug_f))

                    temp = torch.tensor(data_loader.dataset.reducedgrid_shape)
                    proj_mat_aug_f = scene_aug.augment_homography_scene_based(proj_mat_aug_f, [int(x) for x in temp])

                    proj_mat_aug_list.append(torch.linalg.inv(proj_mat_aug_f))


                    # # visualization ###############################################################################################################################
                    # # augment the projection matrix to account for persp aug
                    # temp = torch.tensor([data.shape[-2], data.shape[-1]])
                    # proj_mat_aug = persp_aug.augment_homography_view_based(proj_mat_aug.float(), (temp).float()) # bev-grid reduced -> warped image (720x1280)

                    # # visualize img and projection to bev
                    # img = self.denormalize(data[0,i])
                    # world_feature = kornia.geometry.transform.warp_perspective(img.to('cuda:0'), projm_img2bevred[i].float().to('cuda:0'), data_loader.dataset.reducedgrid_shape)
                    # self.visualize_grid_and_bev(torch.linalg.inv(projm_img2bevred[i]), img, world_feature, os.path.join(f'img_and_bev{i}.jpg'),
                    #                             foot_points * data.shape[-2] / img_gt.shape[-2]) # adjust foot_points for size difference between img and img_gt

                    # # visualize "img features" and projection to bev
                    # resize = torchvision.transforms.Resize((int(img.shape[-2] / data_loader.dataset.img_reduce), int(img.shape[-1] / data_loader.dataset.img_reduce)))
                    # img_resized = resize(img)
                    # world_feature = kornia.geometry.transform.warp_perspective(img_resized.to('cuda:0'),
                    #                                                            projm_imgred2bevred[i].float().to('cuda:0'), data_loader.dataset.reducedgrid_shape)
                    # self.visualize_grid_and_bev(torch.linalg.inv(projm_imgred2bevred[i]), img_resized, world_feature,
                    #                             os.path.join(f'img_and_bev_resized{i}.jpg'),
                    #                             foot_points  * data.shape[-2] / img_gt.shape[-2] / data_loader.dataset.img_reduce) # adjust foot_points for size difference between img and img_gt




                    # visualize augmented image and projection to bev
                    img_aug = self.denormalize(data_aug[0, i])
                    # world_feature = kornia.geometry.transform.warp_perspective(img_aug.to('cuda:0'), torch.linalg.inv(proj_mat_aug).to('cuda:0'), data_loader.dataset.reducedgrid_shape)
                    # self.visualize_grid_and_bev(proj_mat_aug, img_aug, world_feature, os.path.join(f'img_and_bev_aug{i}.jpg'),
                    #                             foot_points_aug * data.shape[-2] / img_gt.shape[-2]) # adjust foot_points for size difference between img and img_gt

                    # visualize augmneted "img features" and projection to bev
                    resize = torchvision.transforms.Resize((int(img_gt.shape[-2] / data_loader.dataset.img_reduce), int(img_gt.shape[-1] / data_loader.dataset.img_reduce)))
                    img_resized = resize(img_aug)
                    temp_mat =  proj_mat_aug_list_without_scene[i]
                    temp_mat_inv = torch.linalg.inv(temp_mat)
                    world_feature = kornia.geometry.transform.warp_perspective(img_resized.to('cuda:0'), temp_mat.to('cuda:0'), data_loader.dataset.reducedgrid_shape)
                    self.visualize_grid_and_bev(temp_mat_inv, img_resized, world_feature,
                                                os.path.join(f'img_and_bev_aug_resized{i}.jpg'),
                                                foot_points_aug / data_loader.dataset.img_reduce,
                                                self.criterion._traget_transform(map_gt, map_gt, data_loader.dataset.map_kernel))
                    
                    # visualize augmneted "img features" and projection to bev, including sene augmentation
                    resize = torchvision.transforms.Resize((int(img_gt.shape[-2] / data_loader.dataset.img_reduce), int(img_gt.shape[-1] / data_loader.dataset.img_reduce)))
                    img_resized = resize(img_aug)
                    temp_mat = proj_mat_aug_list[i]
                    temp_mat_inv = torch.linalg.inv(temp_mat)
                    world_feature = kornia.geometry.transform.warp_perspective(img_resized.to('cuda:0'), temp_mat.to('cuda:0'), data_loader.dataset.reducedgrid_shape)
                    self.visualize_grid_and_bev(temp_mat_inv, img_resized, world_feature,
                                                os.path.join(f'img_and_bev_aug_resized_sceneaug{i}.jpg'),
                                                foot_points_aug / data_loader.dataset.img_reduce,
                                                self.criterion._traget_transform(map_gt_aug, map_gt_aug, data_loader.dataset.map_kernel))
                    



                    # img_label_hm = self.criterion._traget_transform(img_gt, img_gt, data_loader.dataset.img_kernel).cpu().detach().numpy().squeeze()
                    # img_label_hm_head = img_label_hm[0]
                    # img_label_hm_foot = img_label_hm[1]
                    # img0 = Image.fromarray((img.cpu().numpy().squeeze().transpose([1, 2, 0]) * 255).astype('uint8'))
                    # foot_gt_img = add_heatmap_to_image(img_label_hm_foot, img0)
                    # head_gt_img = add_heatmap_to_image(img_label_hm_head, img0)

                    # img_label_hm = self.criterion._traget_transform(img_gt_aug, img_gt_aug, data_loader.dataset.img_kernel).cpu().detach().numpy().squeeze()
                    # img_label_hm_head = img_label_hm[0]
                    # img_label_hm_foot = img_label_hm[1]
                    # img0 = Image.fromarray((img_aug.cpu().numpy().squeeze().transpose([1, 2, 0]) * 255).astype('uint8'))
                    # foot_gt_img_aug = add_heatmap_to_image(img_label_hm_foot, img0)
                    # head_gt_img_aug = add_heatmap_to_image(img_label_hm_head, img0)

                    # fig = plt.figure(figsize=(16,9))
                    # subplt0 = fig.add_subplot(221, title="foot_gt")
                    # subplt1 = fig.add_subplot(222, title="head_gt")                
                    # subplt2 = fig.add_subplot(223, title="foot_gt_aug")                
                    # subplt3 = fig.add_subplot(224, title="head_gt_aug")                
                    # subplt0.imshow(np.array(foot_gt_img))
                    # subplt1.imshow(np.array(head_gt_img))
                    # subplt2.imshow(np.array(foot_gt_img_aug))
                    # subplt3.imshow(np.array(head_gt_img_aug))
                    # plt.savefig(f'ifoot_and_head_gts.jpg', dpi=800)
                    # plt.close(fig)
                    # visualization ###############################################################################################################################


            data, map_gt, imgs_gt, proj_mats = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats_mvaug_features) 
            # TODO if mvaug is not used, the proj_mats in proj_mats_mvaug_features will not be inverted, i.e., subsequent calls to mthe model will not work.  
            # map_res, imgs_res = self.model(data, proj_mats, visualize=True)
            map_res, imgs_res = self.model(data, proj_mats)
            
            t_f = time.time()
            t_forward += t_f - t_b
            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                if not img_gt is None: # may be none after data augmentation
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            loss.backward()
            optimizer.step()
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1), map_res.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, persp_map=False, test_time_aug=False):
        # self.model.configure_model_for_dataset(data_loader.dataset)
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _) in enumerate(data_loader):
            if test_time_aug:
                data, map_gt, imgs_gt, proj_mats = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats)

            with torch.no_grad():
                map_res, imgs_res = self.model(data, proj_mats)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))
                
                # do NMS and create actual preditions (post nms)
                temp = map_grid_res
                scores = temp[temp > self.cls_thres]
                positions = (temp > self.cls_thres).nonzero().float()
                if data_loader.dataset.base.indexing == 'xy':
                    positions = positions[:, [1, 0]]
                else:
                    positions = positions
                if not torch.numel(positions) == 0:
                    ids, count = nms(positions.float(), scores, 20 /  data_loader.dataset.grid_reduce, np.inf)
                    positions = positions[ids[:count], :]
                    scores = scores[ids[:count]]
                map_pseudo_label = torch.zeros_like(map_res)
                for pos in positions:
                    map_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1

            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                if img_gt is not None:
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if visualize:
                for cam_indx, _ in enumerate(imgs_res):
                    cam_number = data_loader.dataset.cameras[cam_indx]

                    pred_view1 = imgs_res[cam_indx]
                    heatmap0_head = pred_view1[0, 0].detach().cpu().numpy().squeeze()
                    heatmap0_foot = pred_view1[0, 1].detach().cpu().numpy().squeeze()

                    if persp_map:
                        map_res_from_perspective = torch.zeros_like(map_res).detach().cpu()
                        map_res_from_perspective_scores = -1e8*torch.ones_like(map_res).detach().cpu()
                        foot_coords = (heatmap0_foot > self.cls_thres).nonzero()
                        foot_scores = heatmap0_foot[heatmap0_foot > self.cls_thres]
                        if not foot_coords[0].size == 0:
                            temp = np.zeros((2, len(foot_coords[0])))
                            temp = np.ones((3, len(foot_coords[0])))
                            temp[0,:] = foot_coords[1]
                            temp[1,:] = foot_coords[0]

                            world_grid = self.model.proj_mats[cam_number] @ temp  
                            world_grid = (world_grid/world_grid[2,:]).detach().cpu().numpy()
                            # temp = temp * data_loader.dataset.img_reduce

                            # print("using projmat ", cam_number)
                            # world_coord = get_worldcoord_from_imagecoord_w_projmat(temp, self.model.proj_mats[cam_number]) # TODO Beware, the proj_mats is a list, not a dict
                            # world_grid = get_worldgrid_from_worldcoord(world_coord)# / data_loader.dataset.grid_reduce
                            for coord_indx, p in enumerate(world_grid.transpose()):
                                if p[0]>=0 and p[1] >= 0 and p[0]<map_res_from_perspective.shape[3] and p[1]<map_res_from_perspective.shape[2]:
                                    map_res_from_perspective[0, 0, int(p[1]), int(p[0])] = 1

                                    prev_val = map_res_from_perspective_scores[0, 0, int(p[1]), int(p[0])]
                                    map_res_from_perspective_scores[0, 0, int(p[1]), int(p[0])] = max(float(foot_scores[coord_indx]), prev_val.item())
                                    # print(p)
                        else:
                            print("No preds from perspective view: ", cam_number+1)

                    img0 = self.denormalize(data[0, cam_indx]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                    # head_cam_result.save(os.path.join(self.logdir, f'output_cam{cam_number+ 1}_head_{batch_idx}.jpg'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'output_cam{cam_number+ 1}_foot_{batch_idx}.jpg'))



                if persp_map:
                    # do NMS and create actual preditions (post nms)
                    perspective_all_res_list = []
                    temp = map_res_from_perspective_scores.squeeze()
                    scores = temp[temp > self.cls_thres].unsqueeze(1)
                    positions = (temp > self.cls_thres).nonzero().float()
                    if data_loader.dataset.base.indexing == 'xy':
                        positions = positions[:, [1, 0]]
                    else:
                        positions = positions

                    perspective_all_res_list.append(torch.cat([torch.ones_like(scores) * frame, positions.float() *
                                                data_loader.dataset.grid_reduce, scores], dim=1))

                    scores = scores.squeeze()
                    if not torch.numel(positions) == 0:
                        ids, count = nms(positions.float(), scores, 20 /  data_loader.dataset.grid_reduce, np.inf)
                        positions = positions[ids[:count], :]
                        scores = scores[ids[:count]]
                    map_perspective_pseudo_label = torch.zeros_like(map_res)
                    for pos in positions:
                        map_perspective_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1


                    fig = plt.figure()
                    subplt0 = fig.add_subplot(321, title="scores")
                    subplt1 = fig.add_subplot(322, title="prediction")
                    subplt2 = fig.add_subplot(323, title="persp. scores")
                    subplt3 = fig.add_subplot(324, title="persp. prediction")
                    subplt4 = fig.add_subplot(325, title="label")
                    subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
                    subplt1.imshow(self.criterion._traget_transform(map_res, map_pseudo_label, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    subplt2.imshow(self.criterion._traget_transform(map_res, map_res_from_perspective, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    subplt3.imshow(self.criterion._traget_transform(map_res, map_perspective_pseudo_label, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    subplt4.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                else:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(321, title="scores")
                    subplt1 = fig.add_subplot(322, title="prediction")
                    subplt4 = fig.add_subplot(323, title="label")

                    subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
                    subplt1.imshow(self.criterion._traget_transform(map_res, map_pseudo_label, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    subplt4.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)




        t1 = time.time()
        t_epoch = t1 - t0

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            # If you want to use the unofiicial python evaluation tool for convenient purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)

            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))
            

        # evaluate perspective view preds
        if persp_map:
            moda = 0
            if res_fpath is not None:
                res_fpath = os.path.abspath(os.path.dirname(res_fpath)) + '/test_perspective.txt'
                all_res_list = torch.cat(perspective_all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res_perspective.txt', all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')

                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.base.__name__)

                # If you want to use the unofiicial python evaluation tool for convenient purposes.
                # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                #                                             data_loader.dataset.base.__name__)
                print("########### perspective results ####################")
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
    

class UDATrainer(BaseTrainer):
    def __init__(self, model, ema_model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0, pom=None,
                 visualize_train=False, target_cameras=None, alpha_teacher=0.99,
                 soft_labels=False, augmentation_module: Augmentation=Augmentation()):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.teacher = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

        # self.pseudo_threshold = 0.7
        self.pom = pom
        self.visualize_train = visualize_train
        self.ema_model = ema_model

        assert target_cameras is not None, "target_cameras must be set in UDATrainer"
        self.target_cameras = target_cameras

        self.alpha_teacher = alpha_teacher
        self.soft_labels = soft_labels

        self.augmentation = augmentation_module

    def train(self, epoch, data_loader, data_loader_target, optimizer, log_interval=100, cyclic_scheduler=None, target_weight=0., pseudo_label_th=0.2):

        self.model.train()
        losses = 0
        losses_target = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, ((data, map_gt, imgs_gt, _, proj_mats_source), (data_target, map_gt_target, imgs_gt_target, _, proj_mats_target)) in enumerate(zip(data_loader, data_loader_target)):

            # train on source data
            optimizer.zero_grad()

            data, map_gt, imgs_gt, proj_mats_source = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats_source)



            map_res, imgs_res = self.model(data, proj_mats_source)
            t_f = time.time()
            t_forward += t_f - t_b
            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            loss.backward()
            # optimizer.step()
            losses += loss.item()

            # logging
            map_res_max = map_res.max()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            img_gt_shape = imgs_gt[0].shape

            if (batch_idx + 1) % log_interval == 0:
                if self.visualize_train:
                    epoch_dir = os.path.join(self.logdir, f'epoch_{epoch}')
                    if not os.path.exists(epoch_dir):
                        os.mkdir(epoch_dir)

                    fig = plt.figure()
                    subplt0 = fig.add_subplot(311, title="student output")
                    subplt1 = fig.add_subplot(312, title="label")
                    subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
                    subplt1.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    plt.savefig(os.path.join(epoch_dir, f'train_source_map_{batch_idx}.jpg'))
                    plt.close(fig)


            del imgs_res, imgs_gt, map_res, map_gt, data

            # train on target data
            # optimizer.zero_grad()

            # create bev pseudo-labels
            with torch.no_grad():
                map_pred_teacher, imgs_teacher_pred = self.ema_model(data_target, proj_mats_target)
            temp = map_pred_teacher.detach().cpu().squeeze()

            if not self.soft_labels:
                scores = temp[temp > pseudo_label_th]
                positions = (temp > pseudo_label_th).nonzero().float()
                if data_loader.dataset.base.indexing == 'xy':
                    positions = positions[:, [1, 0]]
                else:
                    positions = positions
                if not torch.numel(positions) == 0:
                    ids, count = nms(positions.float(), scores, 20 / data_loader.dataset.grid_reduce, np.inf)
                    positions = positions[ids[:count], :]
                    scores = scores[ids[:count]]
                map_pseudo_label = torch.zeros_like(map_pred_teacher)
                for pos in positions:
                    map_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1

                # create perspective view pseudo-labels by projecting bev pseudo-labels into camera
                imgs_pseudo_labels = []
                for cam in self.target_cameras:
                    img_pseudo_label = torch.zeros(img_gt_shape)

                    for grid_pos in positions:
                        pos = data_loader_target.dataset.base.get_pos_from_worldgrid(grid_pos * data_loader_target.dataset.grid_reduce)
                        bbox = self.pom[pos.item()][cam]
                        if bbox is None:
                            continue                    
                        foot_2d = [int((bbox[0] + bbox[2]) / 2), int(bbox[3])]
                        head_2d = [int((bbox[0] + bbox[2]) / 2), int(bbox[1])]
                        img_pseudo_label[:,0,head_2d[1], head_2d[0]] = 1
                        img_pseudo_label[:,1,foot_2d[1],foot_2d[0]] = 1

                    imgs_pseudo_labels.append(img_pseudo_label)

                # apply augmentation to target images and pseudo-labels prior to student training
                data_target, map_pseudo_label, imgs_pseudo_labels, proj_mats_target = self.augmentation.strong_augmentation(data_target,
                                                                                                               map_pseudo_label, imgs_pseudo_labels, proj_mats_target)
                # student predict and compute loss
                map_res_target, imgs_res_target = self.model(data_target, proj_mats_target)
                loss = 0
                for img_res_target, img_pseudo_label in zip(imgs_res_target, imgs_pseudo_labels):
                    if not img_pseudo_label is None:
                        loss += self.criterion(img_res_target, img_pseudo_label.to(img_res_target.device), data_loader_target.dataset.img_kernel)
                loss = self.criterion(map_res_target, map_pseudo_label.to(map_res_target.device), data_loader_target.dataset.map_kernel) + \
                    loss / len([x for x in imgs_pseudo_labels if x is not None]) * self.alpha
            else:
                # apply augmentation to target images and pseudo-labels prior to student training
                map_pseudo_label = map_pred_teacher
                imgs_pseudo_labels = [None]*len(self.target_cameras)
                data_target, map_pseudo_label, imgs_pseudo_labels, proj_mats_target = self.augmentation.strong_augmentation(data_target,
                                                                                                               map_pseudo_label, imgs_pseudo_labels, proj_mats_target)                
                # student predict and compute loss
                map_res_target, imgs_res_target = self.model(data_target, proj_mats_target)
                loss = 0
                loss = self.criterion(map_res_target, map_pseudo_label.to(map_res_target.device), None) # TODO no perspective supervision when using soft-targets?

            # update student
            loss = loss * target_weight # weight the target loss with a weight that grows with increased confidence of pseudo-labels
            loss.backward()
            optimizer.step()
            losses_target += loss.item()

            # update ema model
            alpha_teacher = self.alpha_teacher
            iteration = (epoch - 1) * len(data_loader.dataset) + batch_idx
            self.ema_model = self.update_ema_variables(self.ema_model, self.model, alpha_teacher=alpha_teacher, iteration=iteration)


            # update learning rate
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()

            # logging
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % log_interval == 0:
                if self.visualize_train:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(411, title="student output")
                    subplt1 = fig.add_subplot(412, title="label")
                    subplt2 = fig.add_subplot(413, title="teacher pseudo (or soft) label")
                    subplt3 = fig.add_subplot(414, title="teacher output")
                    subplt0.imshow(map_res_target.cpu().detach().numpy().squeeze())
                    subplt1.imshow(self.criterion._traget_transform(map_res_target, map_gt_target, data_loader_target.dataset.map_kernel)
                                .cpu().detach().numpy().squeeze())
                    if self.soft_labels:
                        subplt2.imshow(self.criterion._traget_transform(map_res_target, map_pseudo_label, None)
                                .cpu().detach().numpy().squeeze())
                    else:
                        subplt2.imshow(self.criterion._traget_transform(map_res_target, map_pseudo_label, data_loader_target.dataset.map_kernel)
                                    .cpu().detach().numpy().squeeze())
                    subplt3.imshow(map_pred_teacher.cpu().detach().numpy().squeeze())
                    epoch_dir = os.path.join(self.logdir, f'epoch_{epoch}')
                    if not os.path.exists(epoch_dir):
                        os.mkdir(epoch_dir)
                    plt.savefig(os.path.join(epoch_dir, f'train_target_map_{batch_idx}.jpg'))
                    plt.close(fig)

                    # visualize pseudo-label of perspective view
                    for cam_indx, img_pseudo_label in enumerate(imgs_pseudo_labels):
                        if img_pseudo_label is None:
                            continue

                        pseudo_view1 = img_pseudo_label
                        pred_view1 = imgs_res_target[cam_indx]
                        pseudo_view1 = self.criterion._traget_transform(pred_view1, pseudo_view1, data_loader_target.dataset.img_kernel).cpu().detach().numpy().squeeze()
                        pseudo_view1_head = pseudo_view1[0]
                        pseudo_view1_foot = pseudo_view1[1]

                        cam_num = self.target_cameras[cam_indx]
                        img0 = self.denormalize(data_target[0, cam_indx]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        img0 = Image.fromarray((img0 * 255).astype('uint8'))
                        # head_cam_result = add_heatmap_to_image(pseudo_view1_head, img0)
                        # head_cam_result.save(os.path.join(epoch_dir, f'head_pseudo_label_cam{cam_num}_{batch_idx}.jpg'))
                        foot_cam_result = add_heatmap_to_image(pseudo_view1_foot, img0)
                        foot_cam_result.save(os.path.join(epoch_dir, f'foot_pseudo_label_cam{cam_num+1}_{batch_idx}.jpg'))

                        # visualizing the heatmap for per-view estimation
                        heatmap0_head = pred_view1[0, 0].detach().cpu().numpy().squeeze()
                        heatmap0_foot = pred_view1[0, 1].detach().cpu().numpy().squeeze()
                        # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                        # head_cam_result.save(os.path.join(epoch_dir, f'output_cam{cam_num+1}_head_{batch_idx}.jpg'))
                        foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                        foot_cam_result.save(os.path.join(epoch_dir, f'student_output_cam{cam_num+1}_foot_{batch_idx}.jpg'))


                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss_source: {:.6f}, Loss_target: {:.6f}, target_weight: {:.2f}'
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), losses_target / (batch_idx + 1), target_weight, precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1), map_res_max))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss_source: {:.6f}, Loss_target: {:.6f},'
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader),losses_target / len(data_loader_target), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats) in enumerate(data_loader):
            with torch.no_grad():
                map_res, imgs_res = self.model(data, proj_mats)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            # head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            # If you want to use the unofiicial python evaluation tool for convenient purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)

            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda
    

    @staticmethod
    def update_ema_variables(ema_model, model, alpha_teacher, iteration):
        # Use the "true" average until the exponential average is more correct
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        # if len(gpus)>1:
        #     for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
        #         #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        #         ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        # else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model



