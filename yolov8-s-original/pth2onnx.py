import torch
import torch.onnx
import os
from nets.yolo import YoloBody


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    
    input_shape     = [640, 640]
    phi             = 's'
    pretrained      = False
    
    model = YoloBody(input_shape, 7, phi, pretrained=pretrained)
    model.load_state_dict(torch.load(checkpoint)) #初始化权重
    model.eval()
    # model.to(device)
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='5'
    checkpoint = './logs/ep080-loss6.296-val_loss4.052.pth'
    onnx_path = './logs/yolos.onnx'
    input = torch.randn(1, 3, 640, 640)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)