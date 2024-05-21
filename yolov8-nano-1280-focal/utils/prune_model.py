import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
# from loguru import logger
import copy
import re 

def count_params(module):
    return sum([p.numel() for p in module.parameters()])

def prune(model, percentage):
    # 计算每个通道的L1-norm并排序
    sorted_channels_list = {}
    prune_model = copy.deepcopy(model)
    device = next(model.parameters()).device
    c1 = 64
    c3 = 256
    for name, module in prune_model.named_modules():
        print(name, end=" ")
        if "backbone" in name or "linear_c" in name:
            print(" ")
            continue
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            # torch.norm用于计算张量的范数,可以计算每个通道上的L1范数 conv.weight.data shape [out_channels,in_channels, k,k]
            if isinstance(module, nn.Conv2d):
                print(module.in_channels,module.out_channels, end=" ")
                module_shape = module.weight.data.shape
                if "conv3" in name:
                    nstr = name.split(".", 1)[0]
                else:
                    nstr = "?"
                
                # 对in_channal维裁剪
                if "linear_fuse" == name or "conv3_for_upsample1.cv1.conv" == name :
                    conv_weight_data = module.weight.data
                    in_channels = module.in_channels
                elif "conv3_for_upsample2.cv1.conv" == name:
                    in_channels = int((module.in_channels-c1) * (1-percentage)) + c1     # 128是backbone的feat1的channal
                    sorted_channels_upsample2cv1=np.concatenate((sorted_channels_old[int(-1*out_channels_old):],range(module.in_channels-c1,module.in_channels)))   
                    conv_weight_data = module.weight.data[:, sorted_channels_upsample2cv1, ...] 
                elif "conv3_for_downsample1.cv1.conv" == name:
                    in_channels = int((module.in_channels) * (1-percentage))
                    sorted_channels_downsample1cv1 = np.concatenate((sorted_channels_old[int(-1*out_channels_old):],sorted_channels_list["conv3_for_upsample1.cv2.conv"][-1*int((in_channels-out_channels_old)):]))
                    conv_weight_data = module.weight.data[:, sorted_channels_downsample1cv1, ...]    
                elif "conv3_for_downsample2.cv1.conv" == name:
                    in_channels = int((module.in_channels-c3) * (1-percentage)) + c3      # 512是backbone的feat3的channal
                    sorted_channels_downsample2cv1=np.concatenate((sorted_channels_old[int(-1*out_channels_old):],range(module.in_channels-c3,module.in_channels)))   
                    conv_weight_data = module.weight.data[:, sorted_channels_downsample2cv1, ...] 
                elif "cv2.0.0.conv" == name or "cv3.0.0.conv" == name:
                    in_channels = int((module.in_channels) * (1-percentage))
                    conv_weight_data = module.weight.data[:,sorted_channels_list["conv3_for_upsample2.cv2.conv"][module.in_channels-in_channels:] , ...]    
                elif "cv2.1.0.conv" == name or "cv3.1.0.conv" == name:
                    in_channels = int((module.in_channels) * (1-percentage))
                    conv_weight_data = module.weight.data[:,sorted_channels_list["conv3_for_downsample1.cv2.conv"][module.in_channels-in_channels:] , ...]    
                    pass
                elif "cv2.2.0.conv" == name or "cv3.2.0.conv" == name:
                    in_channels = int((module.in_channels) * (1-percentage))
                    conv_weight_data = module.weight.data[:,sorted_channels_list["conv3_for_downsample2.cv2.conv"][module.in_channels-in_channels:] , ...]    
                    pass
                elif re.match(r"cv\d.\d.\d.conv",name) or re.match(r"cv\d.\d.\d",name) or "dfl" == name:
                    conv_weight_data = module.weight.data
                    in_channels = module.in_channels
                elif "conv3" in name:     #TODO
                    in_channels = int(module.in_channels * (1-percentage))
                    conv_weight_data = module.weight.data[:, sorted_channels_old[-1*in_channels:], ...]    
                    if "m.0." in name:
                        pass
                    elif "cv1.conv" in name:
                        #TODO
                        pass

                    elif "cv2.conv" in name:
                        sorted_channels_c2fcv2=np.concatenate((sorted_channels_list[nstr+".cv1.conv"],sorted_channels_old[int(-1*out_channels_old):]))
                        conv_weight_data = module.weight.data[:, sorted_channels_c2fcv2, ...]    
                    else:
                        pass

                else:
                    in_channels = int(module.in_channels * (1-percentage))
                    conv_weight_data = module.weight.data[:, sorted_channels_old[-1*in_channels:], ...]

                # 对通道进行排序,返回索引
                importance_conv = torch.norm(conv_weight_data, 1, dim=(1, 2, 3))
                
                # 对out_channal维裁剪
                if "linear_pred.2" in name or re.match(r"cv\d.\d.\d.conv",name) or re.match(r"cv\d.\d.\d",name) or "dfl" == name:
                    out_channels=module.out_channels  #  最后一层不变，sorted_channels也得操作一下
                    sorted_channels_list[name] = range(out_channels)    # 最后一层不变sorted_channels
                    sorted_channels = sorted_channels_list[name]
                elif re.match(r"conv3_for_(downsample\d|upsample\d).cv1.conv",name):
                    out_channels = int(module.out_channels * (1-percentage))
                    importance_conv1 = importance_conv[:int(module.out_channels*0.5),...]
                    importance_conv2 = importance_conv[int(module.out_channels*0.5):,...]
                    sorted_channels1 = np.argsort(np.concatenate([x.cpu().numpy().flatten() for x in importance_conv1]))  # 0.5用的是C2F的默认值
                    sorted_channels2 = np.argsort(np.concatenate([x.cpu().numpy().flatten() for x in importance_conv2]))
                    sorted_channels_list[name] = np.concatenate((sorted_channels1[int(-1*out_channels*percentage):],sorted_channels2[int(-1*out_channels*percentage):]))
                    sorted_channels = sorted_channels_list[name]  
                else:
                    out_channels = int(module.out_channels * (1-percentage))
                    sorted_channels_list[name] = np.argsort(np.concatenate([x.cpu().numpy().flatten() for x in importance_conv]))
                    sorted_channels = sorted_channels_list[name]
                    pass
                if re.match(r"conv3_for_(downsample\d|upsample\d).cv2.conv",name):   # c2f 剪枝后要改c的值
                    # setattr(model, nstr+".c", out_channels * 0.5)                    # 0.5用的是C2F的默认值
                    _set_module(model, nstr+".c", int(out_channels * 0.5))
                new_module = nn.Conv2d(in_channels,     # 因为第一层是3最后一层是4
                                        out_channels,
                                        kernel_size=module.kernel_size,
                                        stride=module.stride,
                                        padding=module.padding,
                                        dilation=module.dilation,
                                        groups=module.groups,
                                        bias=(module.bias is not None)
                                        ).to(device)
                # in_channels = new_module.out_channels  # 因为前一层的输出通道会影响下一层的输入通道
                # 重新分配权重 权重的shape[out_channels, in_channels, k, k]
                # c2, c1, _, _ = new_module.weight.data.shape
                # print(new_module.weight.data.shape)
                # new_module.weight.data[...] = module.weight.data[num_channels_to_prune:, :c1, ...]
                new_module.weight.data[...] = conv_weight_data[sorted_channels[-1*out_channels:], :,...]      # todo  输入层先不管他
                if module.bias is not None:
                    new_module.bias.data[...] = module.bias.data[sorted_channels[-1*out_channels:]]
                # 用新卷积替换旧卷积
                # setattr(model, name, new_module)
                _set_module(model, f"{name}", new_module)
                new_module_shape = new_module.weight.data.shape
                print(new_module_shape[1],new_module_shape[0], end=" ")
                sorted_channels_old = sorted_channels
                out_channels_old = out_channels
            elif isinstance(module, nn.BatchNorm2d):
                print(module.num_features, end=" ")
                new_bn = nn.BatchNorm2d(num_features=new_module.out_channels,
                                        eps=module.eps,
                                        momentum=module.momentum,
                                        affine=module.affine,
                                        track_running_stats=module.track_running_stats).to(next(model.parameters()).device)
                new_bn.weight.data[...] = module.weight.data[sorted_channels_old[-1*out_channels:]]
                if module.bias is not None:
                    new_bn.bias.data[...] = module.bias.data[sorted_channels_old[-1*out_channels:]]
                # 用新bn替换旧bn
                # setattr(model, name, new_bn)
                _set_module(model, f"{name}", new_bn)
                print(new_bn.num_features, end=" ")
        print(" ")
        if"cv3.2.2" == name:    # 结束了
            break
    # return model

# 核心函数，参考了torch.quantization.fuse_modules()的实现
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

# def plot_weights(model, layer_name):
#     for name, param in model.named_parameters():
#         if name in layer_name:
#             plt.figure()
#             plt.title(name)
#             plt.xlabel('Values')
#             plt.ylabel('Frequency')
#             values = param.cpu().data.numpy().flatten()
#             mean = np.mean(values)
#             std = np.std(values)
#             plt.text(0.95, 0.95, 'Mean:{:.2f}\nStd: {:.2f}'.format(mean, std), transform=plt.gca().transAxes,
#                      ha='right',
#                      va='top')
#             sns.histplot(values, kde=False, bins=50)
#             plt.show()

# def plot_3D_weights(model, layer_name):
#     for name, param in model.named_parameters():
#         print(name)
#         if name in layer_name:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             values = param.cpu().data.numpy().flatten()
#             # x对应输出通道，y对应输入通道
#             x, y, z = np.indices((param.shape[0], param.shape[1], param.shape[2] * param.shape[3]))
#             ax.scatter(x, y, z, c=values, cmap='jet')
#             fig.colorbar(ax.get_children()[0], ax=ax)
#             ax.set_xlabel('out_channels')
#             ax.set_ylabel('in_channels')
#             ax.set_zlabel('values')
#             plt.title(name + ' weights distribution')
#             plt.show()
