# DAC2024-IMSCLAB

For DAC2024

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
1.move to ./based_on_yolov8-pytorch-master
2.run .py
```
python voc_annotation.py
```
then you will get 2007_train.txt, 2007_val.txt, seg_train.txt, seg_val.txt

third, run train.py
```
python train.py
```



