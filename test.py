import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])



    if 'wildtrack' in args.dataset:
        # data_path = os.path.expanduser('/data/Wildtrack')
        assert args.data_path is not None, "must specify data path"
        data_path = args.data_path
        if args.cam_adapt:
            assert args.trg_cams is not None, "trg_cams must be specified in cam_adapt setting"
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]
            print("trg_cams: ", trg_cams)
            test_base = Wildtrack(data_path, cameras=trg_cams)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = Wildtrack(data_path, cameras=trg_cams)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
        else:
            test_base = Wildtrack(data_path)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)            
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = Wildtrack(data_path)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)

    else:
        raise Exception('must choose from [wildtrack]')



    # model
    if args.variant == 'default':
        model = PerspTransDetector(test_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(test_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(test_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(test_set, args.arch)
    else:
        raise Exception('no support for this variant')

    # loss
    criterion = GaussianMSE().cuda()

    # logging
    idx = 0
    while os.path.exists(os.path.join(args.log_dir, "test_"+str(idx))):
        idx += 1

    logdir=os.path.join(args.log_dir, "test_"+str(idx))
    os.makedirs(logdir, exist_ok=True)

    sys.stdout = Logger(os.path.join(logdir, 'test_log.txt'), )
    print('Settings:')
    print(vars(args))

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)

    # learn
    resume_fname = os.path.join(args.log_dir, args.model)
    print("Loading saved model from: ", resume_fname)
    model.load_state_dict(torch.load(resume_fname))

    print('Testing...')
    if args.train_set:
        trainer.test(train_loader, os.path.join(logdir, 'test.txt'), test_set.gt_fpath, True)
    else:
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), test_set.gt_fpath, True)

if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')

    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--cam_adapt', action="store_true")
    parser.add_argument('--train_set', action="store_true")
    parser.add_argument('--trg_cams', type=str, default=None)
    parser.add_argument('--model', type=str, default="MultiviewDetector.pth")

    args = parser.parse_args()


    main(args)
