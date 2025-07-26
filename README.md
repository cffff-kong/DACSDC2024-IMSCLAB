# DAC2024-IMSCLAB
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
For DAC2024
最终会用到的文件夹有三个：yolov8-nano-1280-focal,other_code,dac_sdc


2024.6.10
今天对 onnx 版本做最后一次修订，修订的内容在 dac_sdc 中，在这个版本里修改了之前的两处错误，已经可以正常使用了，两处错误分别是：首先是官方定义的 xy 方向与我们不同，导致 det 任务出现了问题，解决方法是把 xy 换过来就好了；之后是对于 segmentation 任务我们输出的是一个大小为 1280 1280 只包含 012 的灰度图，由于原始图像形状并不是正方形，所以我们在将 1280 1280 还原到原始大小的时候需要先恢复比例。  



2024.5.5
channal pruning v1
没有裁剪 backbone 和最后的 cv2cv3 的输出 channal，中间裁减了一下
train.py 和 predict.py 均有 is_prune 属性，置 True 就开始了。
暂时可能似乎只有 0.5 好用，四舍五入时候也许会出问题（我也不确定）
channal 的对应顺序没有严格 check，大体上没有问题

2024.4.29
新增两个 nano 文件夹，其中数据集和 logs 文件夹删除，需要重新在 root 下创建一个 logs 文件夹，再把 voc2007 复制进对应路径，之后运行 voc_annotations.py 即可。此外共享链接里面是两个 pretrain model 和两个 train 后的目前最好版本。

```
https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jklee_connect_hku_hk/EoPtJkYYUwFNhH54T_NyOFQBkwYa_jLI2xg0bGCPvfTWXQ?e=N0Y8tQ
```

2024.4.25
新增 DAC-pure，去掉了打分等功能，实现了仅 seg+det 的效果，用作基准版，后续添加功能前先在该版本测试

2024.4.15
A new folder has been added that only adds the HEAD of the segmentation task, but the LOSS is not involved in the calculation, and the codes in this folder can be used to verify where the problem is occurring.

2024.4.15
还是写中文了，看得清楚些。
我把 seg 的数据集加入到了总的 dataloader 里面，然后我把 mosaic 数据增强整个删掉了，然后发现模型可以用了，就很神奇。

for based_on_yolov8-pytorch-master

dataset root:
you need to upload the dataset, replace Annotations, JPEGImages and label file loader.
build ImageSets/Main file loader.

```
├─VOCdevit/VOC2007
│ ├─Annotations
│ │ ├─.xml
│ ├─ImageSets
│ │ ├─Main
│ │ │ ├─test.txt
│ │ │ ├─train.txt
│ │ │ ├─trainval.txt
│ │ │ ├─val.txt
│ ├─JPEGImages
│ │ ├─.jpg
│ ├─label
│ │ ├─.json
│ ├─seg
│ │ ├─.png
```

first, you need to generate the dataset.
1.move to ./VOC2007
2.run .py in order

```
python del_Annotations.py
```

```
python del_imgs.py
```

```
python gen_seg.py
```

second, you need generate the train txt
1.move to root
2.run .py

```
python voc_annotation.py
```

then you will get 2007_train.txt, 2007_val.txt, seg_train.txt, seg_val.txt

third, run train.py

```
python train.py
```

about how to predict

find the training model in /log, then modified the yolo.py( "model_path" ) in the root file loader.

```
python predict.py
```

input the path of image.
