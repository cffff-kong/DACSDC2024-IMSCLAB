# DAC2024-IMSCLAB

For DAC2024


YOLOX-nano:

train:
```
python tools/train.py -f exps/example/yolox_voc/yolox_nano.py -d 1 -b 32 -c weights/yolox_nano.pth
```
predict:
```
python tools/demo.py image -f exps/example/yolox_voc/yolox_nano.py -c /home/RRAM_HKU/YOLOX/YOLOX_outputs/yolox_nano/best_ckpt.pth --path /home/RRAM_HKU/YOLOX/VOCdevkit/VOC2007/JPEGImages/07923.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```




nanodet：
train:
```
python ./tools/train.py ./config/my_dataset.yml
```

pridict:
use predict.ipynb in /nanodet-main/
