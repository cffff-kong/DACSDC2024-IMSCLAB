val_file = "../VOCdevkit/VOC2007/ImageSets/Main/val.txt"
train_file = "../VOCdevkit/VOC2007/ImageSets/Main/train.txt"

# 读取val.txt中的内容，并提取数字去掉下划线后的内容
with open(val_file, 'r') as f:
    val_content = [line.strip().split('_')[0] for line in f.readlines()]

# 读取train.txt中的内容，并提取数字去掉下划线后的内容
with open(train_file, 'r') as f:
    train_content = [line.strip().split('_')[0] for line in f.readlines()]

# 将提取的内容写入val_new.txt
val_new_file = "../VOCdevkit/VOC2007/ImageSets/Main/val_new.txt"
with open(val_new_file, 'w') as f:
    f.write('\n'.join(val_content))

# 将提取的内容写入train_new.txt
train_new_file = "../VOCdevkit/VOC2007/ImageSets/Main/train_new.txt"
with open(train_new_file, 'w') as f:
    f.write('\n'.join(train_content))

# 检查val_new.txt和train_new.txt中的内容，如果存在重复，则将val.txt中对应行的内容删除
with open(val_new_file, 'r') as f:
    val_new_content = set(line.strip() for line in f)

with open(train_new_file, 'r') as f:
    train_new_content = set(line.strip() for line in f)

# 找出val.txt中需要删除的行的内容
lines_to_remove = []
with open(val_file, 'r') as f:
    for line in f:
        line = line.strip().split('_')[0]
        if line in train_new_content:
            lines_to_remove.append(line)

# 从val.txt中删除需要删除的行的内容，并删除所有空白行
with open(val_file, 'r+') as f:
    lines = f.readlines()
    f.seek(0)
    for line in lines:
        line = line.strip().split('_')[0]
        if line not in lines_to_remove and line:
            f.write(line + '\n')
    f.truncate()
