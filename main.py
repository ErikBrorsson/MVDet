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
from multiview_detector.trainer import PerspectiveTrainer, UDATrainer, Augmentation


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
        data_path = args.data_path
        if args.cam_adapt:
            assert args.src_cams is not None and args.trg_cams is not None, "src_cams and trg_cams must be specified in cam_adapt setting"
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]

            src_cams = args.src_cams.split(",")
            src_cams = [int(x) for x in src_cams]

            source_base = Wildtrack(data_path, cameras=src_cams)
            target_base = Wildtrack(data_path, cameras=trg_cams)
            test_base = Wildtrack(data_path, cameras=trg_cams)

            train_set = frameDataset(source_base, train=True, transform=train_trans, grid_reduce=4)
            train_set_target = frameDataset(target_base, train=True, transform=train_trans, grid_reduce=4)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True)
            train_loader_target = torch.utils.data.DataLoader(train_set_target, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
        else:
            base = Wildtrack(data_path)
            test_base = base

            train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)

    else:
        raise Exception('must choose from [wildtrack]')



    # model
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch, pretrained=args.pretrained)

        if args.uda:
            # init ema model
            ema_model = PerspTransDetector(train_set, args.arch, pretrained=args.pretrained)
            for param in ema_model.parameters():
                param.detach_()
            mp = list(model.parameters())
            mcp = list(ema_model.parameters())
            n = len(mp)
            for i in range(0, n):
                mcp[i].data[:] = mp[i].data[:].clone()

    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    else:
        raise Exception('no support for this variant')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = GaussianMSE().cuda()

    # logging
    logdir = f'logs/{args.dataset}_frame/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')# if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
    if args.log_dir is not None:
        logdir = os.path.join(args.log_dir, logdir)

    # if args.resume is None:
    os.makedirs(logdir, exist_ok=True)

    # copying files like this gave rise to issues on the cluster when running many experiments simultaneously
    # copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
    # for script in os.listdir('.'):
    #     if script.split('.')[-1] == 'py':
    #         dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
    #         shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    for k, v in vars(args).items():
        print(k, ": ", v)

    print("logdir: ", logdir)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    augmentation = Augmentation(args.dropview, args.permutation, args.mvaug)

    if args.uda:
        pom = train_loader_target.dataset.base.read_pom()
        trainer = UDATrainer(model, ema_model, criterion, logdir, denormalize, args.cls_thres, args.alpha, pom,
                             args.train_viz, target_cameras=target_base.cameras,
                             alpha_teacher=args.alpha_teacher, soft_labels=args.soft_labels,
                             augmentation_module=augmentation)
    else:
        trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha, augmentation)

    # learn
    if args.resume_model is not None:
        # resume_dir = f'logs/{args.dataset}_frame/{args.variant}/' + args.resume
        # resume_fname = resume_dir + '/MultiviewDetector.pth'
        resume_fname = args.resume_model
        print("Loading saved model from: ", resume_fname)
        model.load_state_dict(torch.load(resume_fname))


    # print('Testing...')
    # trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)

    if args.uda:
        if args.target_epoch_start is None or args.target_weight_start is None or args.target_weight_end is None:
            # randomize the target weight schedule
            # target_epoch_start = np.random.choice(10) + 1
            target_epoch_start = np.random.choice(7) + 4
            target_weight_start = np.random.rand()
            target_weight_end = target_weight_start + (1- target_weight_start)*np.random.rand()
        else:
            target_epoch_start = args.target_epoch_start
            target_weight_start = args.target_weight_start
            target_weight_end = args.target_weight_end
        
        target_weights = [0. for x in range(10)]
        increment_steps = args.epochs - target_epoch_start
        if increment_steps == 0:
            step_size = 0
        else:
            step_size = (target_weight_end - target_weight_start) / increment_steps
        for i in range(increment_steps + 1):
            target_weights[i + target_epoch_start - 1] = target_weight_start + step_size * i

        print("target_epoch_start: ", target_epoch_start)
        print("target_weight_start: ", target_weight_start)
        print("target_weight_end: ", target_weight_end)
        print("target_weights: ", target_weights)

        if args.pseudo_label_th is None:
            pseudo_label_th = 0.37 + np.random.rand()*0.5 # random value between 0.3 and 0.45
        else:
            pseudo_label_th = args.pseudo_label_th
        print("pseudo_label_th: ", pseudo_label_th)

    # print('Testing...')
    # test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
    #                                             test_set.gt_fpath, True)
    max_moda = -1e10
    best_epoch = -1
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        print('Training...')
        if args.uda:
            target_weight = target_weights[epoch - 1]
            train_loss, train_prec = trainer.train(epoch, train_loader, train_loader_target, optimizer, args.log_interval, scheduler,target_weight,pseudo_label_th)
        else:
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
        print('Testing...')
        test_loss, test_prec, moda, modp, precision, recall = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                    train_set.gt_fpath, True)

        if moda >= max_moda:
            max_modp, max_precision, max_recall = modp, precision, recall
            max_moda = moda
            best_epoch = epoch
            # save model after every epoch
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
            if args.uda:
                torch.save(ema_model.state_dict(), os.path.join(logdir, 'MultiviewDetector_ema.pth'))


        x_epoch.append(epoch)
        train_loss_s.append(train_loss)
        train_prec_s.append(train_prec)
        test_loss_s.append(test_loss)
        test_prec_s.append(test_prec)
        test_moda_s.append(moda)
        draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                    test_loss_s, test_prec_s, test_moda_s)
        
        print('max_moda: {:.1f}%, max_modp: {:.1f}%, max_precision: {:.1f}%, max_recall: {:.1f}%, epoch: {:.1f}%'.
                format(max_moda, max_modp, max_precision, max_recall, best_epoch))
        

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
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--train_viz', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--cam_adapt', action="store_true")
    parser.add_argument('--uda', action="store_true")
    parser.add_argument('--dropview', action="store_true")
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--mvaug", action="store_true")
    parser.add_argument('--soft_labels', action="store_true")
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--src_cams', type=str, default=None)
    parser.add_argument('--trg_cams', type=str, default=None)
    parser.add_argument('--alpha_teacher', type=float, default=0.99)

    # below parameters are randomized if not set
    parser.add_argument('--target_epoch_start', type=int, default=None, help='the epoch at which training on target domain starts')
    parser.add_argument('--target_weight_start', type=float, default=None, help='the initial weight when training on target domain starts')
    parser.add_argument('--target_weight_end', type=float, default=None, help='the final weight when training on target domain ends')
    parser.add_argument('--pseudo_label_th', type=float, default=None, help='confidenbce threshold for creating pseudo-labels')


    args = parser.parse_args()

    if args.config is not None:
        import json
        f = open(args.config)
        data = json.load(f)
        args_d = vars(args)

        for k,v in data.items():
            args_d[k] = v

    main(args)
