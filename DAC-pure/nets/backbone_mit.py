import torch
import torch.nn as nn

from nets.mit import mit_b0

from mmengine.runner import load_checkpoint



class yolo_mit(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = mit_b0()
        



    def forward(self, x):
        conv1, conv2, conv3, conv4 = self.backbone(x)

        # return conv1, conv2, conv3, conv4
        return conv2, conv3, conv4





    
    