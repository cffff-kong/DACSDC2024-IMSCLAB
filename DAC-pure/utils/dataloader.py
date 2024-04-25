from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, seg_lines, input_shape, num_classes, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.seg_lines          = seg_lines 
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#

        image, mask, box      = self.get_random_data(self.annotation_lines[index], self.seg_lines[index], self.input_shape, random = self.train)
        mask[mask == 255] = 0  
        mask[mask == 8] = 1
        mask[mask == 9] = 2
        mask[mask == 10] = 3
        masks = np.expand_dims(mask, axis=-1)
        masks = masks.transpose((2, 0, 1))

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        
        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL          = len(box)
        labels_out  = np.zeros((nL, 6))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            
            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            #---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
            
        return image, masks, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, seg_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        image_path, mask_path = seg_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        mask = Image.open(mask_path).convert("L")
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        #---------------------------------#
        #   将图像多余的部分加上灰条
        #---------------------------------#
        image       = image.resize((nw,nh), Image.BICUBIC)
        mask        = mask.resize((nw, nh), Image.NEAREST) 
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        new_mask    = Image.new('L', (w, h), (255))
        new_mask.paste(mask, (dx, dy))
        image_data  = np.array(new_image, np.float32)
        mask_data   = np.array(new_mask, np.float32)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, mask_data, box
                
        
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    
    
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    masks  = []
    for i, (img, mask, box) in enumerate(batch):
        images.append(img)
        masks.append(mask)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    masks  = torch.from_numpy(np.array(masks)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, masks, bboxes

# # DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images      = []
#     n_max_boxes = 0
#     bs          = len(batch)
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         n_max_boxes = max(n_max_boxes, len(box))
    
#     bboxes  = torch.zeros((bs, n_max_boxes, 4))
#     labels  = torch.zeros((bs, n_max_boxes, 1))
#     masks   = torch.zeros((bs, n_max_boxes, 1))
    
#     for i, (img, box) in enumerate(batch):
#         _sub_length = len(box)
#         bboxes[i, :_sub_length] = box[:, :4]
#         labels[i, :_sub_length] = box[:, 4]
#         masks[i, :_sub_length]  = 1
    
#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes, labels, masks
