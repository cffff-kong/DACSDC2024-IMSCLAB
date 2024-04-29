# DAC2024-IMSCLAB

For DAC2024

2024.4.29
新增两个nano文件夹，其中数据集和logs文件夹删除，需要重新在root下创建一个logs文件夹，再把voc2007复制进对应路径，之后运行voc_annotations.py即可。此外共享链接里面是两个pretrain model和两个train后的目前最好版本。
```
https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jklee_connect_hku_hk/EoPtJkYYUwFNhH54T_NyOFQBkwYa_jLI2xg0bGCPvfTWXQ?e=N0Y8tQ
```
2024.4.25
新增DAC-pure，去掉了打分等功能，实现了仅seg+det的效果，用作基准版，后续添加功能前先在该版本测试

2024.4.15
A new folder has been added that only adds the HEAD of the segmentation task, but the LOSS is not involved in the calculation, and the codes in this folder can be used to verify where the problem is occurring.

2024.4.15
还是写中文了，看得清楚些。
我把seg的数据集加入到了总的dataloader里面，然后我把mosaic数据增强整个删掉了，然后发现模型可以用了，就很神奇。


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

