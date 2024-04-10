import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utils.utils import get_lr
        


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, labels, labels_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    seg_loss = torch.nn.CrossEntropyLoss()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()

    


    for iteration, (yolo_batch1, seg_batch1) in enumerate(zip(gen, labels)):
        if iteration >= epoch_step:
            break

        images, bboxes = yolo_batch1
        image, label = seg_batch1


        # print("label shape:", label.shape)
        
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
                label  = label.cuda(local_rank) 
                # print(label.size())
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            # dbox, cls, origin_cls, anchors, strides 
            outputs = model_train(images)

            
            #################################################################我加的
            # print(f'outputs:{outputs[1].size()}')

            # # 获取三个通道（channel）的数据
            # channel_data = outputs[1][0, :3, :, :].cpu()  # 选择前三个通道的数据

            # # 转换为NumPy数组
            # numpy_array = channel_data.detach().numpy()

            # # 使用Matplotlib绘制图像
            # plt.imshow(numpy_array.transpose(1, 2, 0))  # 转置通道顺序为RGB
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
            # plt.savefig('/home/RRAM_HKU/yolo/yolov8-pytorch-master/yolov8-pytorch-master/img/1.png')

            # label = Image.open('/home/RRAM_HKU/yolo/yolov8-pytorch-master/yolov8-pytorch-master/VOCdevkit/VOC2007/seg/00001.png')
            # label = label.resize((640, 640), Image.NEAREST)
            # #label变成tensor且通道数变成[32, 3, 640, 640]
            # label = torch.tensor(np.array(label)).unsqueeze(0).unsqueeze(0)
            # label = label.repeat(1, 3, 1, 1)
            # label = label.cuda(local_rank)
            # #变成batchsize=32
            # label = label.repeat(32, 1, 1, 1)
            # # print(f'label:{label.size()}')
            #################################################################我加的

            

            loss_det = yolo_loss(outputs[0], bboxes)
            
            loss_seg = seg_loss(outputs[1], label)
            
            # dbox, cls, origin_cls, anchors, strides, seg = model_train(images)
            # loss_det = yolo_loss(dbox, cls, origin_cls, anchors, strides, bboxes)
            # loss_seg = seg_loss(seg, label)

            loss_value = loss_seg + loss_det
            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, (yolo_batch2, seg_batch2) in enumerate(zip(gen_val, labels_val)):
        if iteration >= epoch_step_val:
            break
        images, bboxes = yolo_batch2
        image, label = seg_batch2
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
                label = label.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train_eval(images)
            loss_det = yolo_loss(outputs[0], bboxes)
            
            loss_seg = seg_loss(outputs[1], label)

            loss_value = loss_seg + loss_det

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))