import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torchvision

from models.model import GLPDepth
from models.model_seg_ori_icnet import GLPDepth_seg
import utils.metrics as metrics
from utils.criterion import SiLogLoss
import utils.logging as logging

from dataset.base_dataset import get_dataset
from dataset.nyudseg import NYUDSegDataset
from configs.train_options import TrainOptions
from dataset.nyud_single_seg import NYUD_SINGLE_SEG
from dataset import transforms

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def scores(label_trues, label_preds, num_classes=40):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    #print(hist.sum(axis=1))
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    #cls_iu = dict(zip(range(num_classes), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        #"Class IoU": cls_iu,
    }

def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Logging
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    log_txt = os.path.join(log_dir, 'logs_seg_icnet_lossd5_small_patch_supervision.txt')  
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = GLPDepth_seg(max_depth=args.max_depth, is_train=True)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    # train_dataset = NYUDSegDataset(
    #     root_dir='/home/zijian/projects/depth/GLPDepth/datasets/nyud_seg',
    #     split='training',
    #     stage='train',
    #     aug=True,
    #     resize_range= [512, 2048],
    #     rescale_range= [0.5, 2.0],
    #     crop_size= 512,
    #     img_fliplr=True,
    #     ignore_index= 255,
    #     num_classes= 40,
    # )
    
    # val_dataset = NYUDSegDataset(
    #     root_dir='/home/zijian/projects/depth/GLPDepth/datasets/nyud_seg',
    #     split='validation',
    #     stage='val',
    #     aug=False,
    #     # resize_range= [512, 2048],
    #     # rescale_range= [0.5, 2.0],
    #     # crop_size= 512,
    #     # img_fliplr=True,
    #     ignore_index= 255,
    #     num_classes= 40,
    # )

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                            shuffle=True, num_workers=args.workers, 
    #                                            pin_memory=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
    #                                          pin_memory=True)
    
    train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=(448, 576), cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=(448, 576)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

    # Testing 
    valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=(448, 576)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
    
    train_dataset = NYUD_SINGLE_SEG("/mnt/mainHD/zijian/projects/depth/GLPDepth/NYUDv2", download=False, split='train', transform=train_transforms, 
                                    do_semseg=True,  overfit=False)
    
    val_dataset = NYUD_SINGLE_SEG("/mnt/mainHD/zijian/projects/depth/GLPDepth/NYUDv2", download=False, split='val', transform=valid_transforms, 
                                    do_semseg=True,  overfit=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True,drop_last=False)

    
        
    # Training settings
    criterion_d = torch.nn.CrossEntropyLoss(ignore_index = 255)
    #criterion_d = torch.nn.CrossEntropyLoss()
    criterion_d = criterion_d.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    global global_step
    global_step = 0

    # Perform experiment
    for epoch in range(1, args.epochs +1 ):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        writer.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                              device=device, epoch=epoch, args=args,
                                              log_dir=log_dir)
            writer.add_scalar('Val loss', loss_val, epoch)
            #print(results_dict)
            result_lines = logging.display_result(results_dict)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)                

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)
                
            # filename = './seg_model/model_checkpoint_epoch_%03d.pth' % epoch
            # torch.save(model.state_dict(), filename)


def train(train_loader, model, criterion_d, optimizer, device, epoch, args):    
    global global_step
    model.train()
    seg_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2

    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        for param_group in optimizer.param_groups:
            if global_step < 2019 * half_epoch:
                current_lr = (1e-4 - 3e-5) * (global_step /
                                              2019/half_epoch) ** 0.9 + 3e-5
            else:
                current_lr = (3e-5 - 1e-4) * (global_step /
                                              2019/half_epoch - 1) ** 0.9 + 1e-4
            param_group['lr'] = current_lr
        #print(batch)
        input_RGB = batch['image'].to(device)
        depth_gt = batch['semseg'].to(device)
        
        #preds = model(input_RGB)
        
        downsampled_input = F.interpolate(input_RGB, scale_factor=0.5, mode='bilinear', align_corners=False).to(device)
        
        preds1, preds2, preds3, preds4, preds5 = model(downsampled_input)
        # print(preds1.shape)
        # print(preds2.shape)
        # print(preds3.shape)
        # print(preds4.shape)       
        # print(preds5.shape)

        
        optimizer.zero_grad()
        #preds = F.interpolate(preds, size=depth_gt.shape[1:], mode='bilinear', align_corners=False)
        #preds = preds.squeeze(1)
        #depth_gt = depth_gt.squeeze(1)
        #depth_gt = depth_gt.unsqueeze(1)

        downsampled_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest').to(device)

        # downsampled_gt2 = F.interpolate(depth_gt, scale_factor=0.0625, mode='nearest').to(device)
        # downsampled_gt3 = F.interpolate(depth_gt, scale_factor=0.03125, mode='nearest').to(device)
        # downsampled_gt4 = F.interpolate(depth_gt, scale_factor=0.015625, mode='nearest').to(device)
        #preds5 = F.interpolate(preds5, scale_factor=2, mode='bilinear', align_corners=False).to(device)
        
        # print(downsampled_gt1.shape)
        # print(downsampled_gt2.shape)
        # print(downsampled_gt3)
        # print(downsampled_gt4)
        # print(preds5.shape)
    
        #print(preds.size())
        #print(depth_gt)

        loss_d5 = criterion_d(preds5, downsampled_gt.squeeze(1).type(torch.long))
        #print(loss_d5)
        # loss_d4 = criterion_d(preds4, downsampled_gt4.squeeze(1).type(torch.long))

        # loss_d3 = criterion_d(preds3, downsampled_gt3.squeeze(1).type(torch.long))
        # loss_d2 = criterion_d(preds2, downsampled_gt2.squeeze(1).type(torch.long))
        # loss_d1 = criterion_d(preds1, downsampled_gt1.squeeze(1).type(torch.long))
        #loss_d = loss_d1 + loss_d2 + loss_d3 + loss_d4 + loss_d5
        loss_d = loss_d5
        #seg_loss.update(loss_d.item(), downsampled_input.size(0))
        #print(loss_d)
        loss_d.backward()
        # print('7')

        # logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
        #                     ('Depth Loss: %.4f (%.4f)' %
        #                     (seg_loss.val, seg_loss.avg)))
        
        logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %.4f ' %
                            (loss_d)))
        optimizer.step()

    return loss_d


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir):
    depth_loss = logging.AverageMeter()
    model.eval()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            log_dir, 'epoch_%02d_model.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    predlist, gts = [], []
    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['semseg'].to(device)
        #filename = batch['filename'][0]
        
        downsampled_input = F.interpolate(input_RGB, scale_factor=0.5, mode='bilinear', align_corners=False)

        with torch.no_grad():
            preds1, preds2, preds3, preds4, preds = model(downsampled_input)
        # print(preds.shape)
        # print(depth_gt.shape)
        # pred_d = preds.squeeze()
        # depth_gt = depth_gt.squeeze().long()
        pred_d = preds
        depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest')
        depth_gt = depth_gt.long().squeeze(1)

        loss_d = criterion_d(pred_d, depth_gt)

        depth_loss.update(loss_d.item(), input_RGB.size(0))
        

        predlist += list(
                torch.argmax(pred_d,
                             dim=1).cpu().numpy().astype(np.int16))
        gts += list(depth_gt.cpu().numpy().astype(np.int16))

        # pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        # computed_result = metrics.eval_depth(pred_crop, gt_crop)
        # save_path = os.path.join(result_dir, filename)

        # if save_path.split('.')[-1] == 'jpg':
        #     save_path = save_path.replace('jpg', 'png')

        # if args.save_result:
        #     if args.dataset == 'kitti':
        #         pred_d_numpy = pred_d.cpu().numpy() * 256.0
        #         cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #     else:
        #         pred_d_numpy = pred_d.cpu().numpy() * 1000.0
        #         cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

    #     for key in result_metrics.keys():
    #         result_metrics[key] += computed_result[key]

    # for key in result_metrics.keys():
    #     result_metrics[key] = result_metrics[key] / (batch_idx + 1)
    score = scores(gts, predlist)
    return score, loss_d


if __name__ == '__main__':
    main()
