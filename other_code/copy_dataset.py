import os
import shutil
import xml.etree.ElementTree as ET

# 定义文件路径
xml_dir = './VOC2007/Annotations/'
img_dir = './VOC2007/JPEGImages/'
json_dir = './VOC2007/label/'
seg_dir = './VOC2007/seg/'

processed_images_yellow_light = 0
processed_images_off_light = 0
processed_images_Light_Red = 0
processed_images_Light_green = 0
processed_images_Pedestrian = 0

# 遍历xml文件
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith('.xml'):
        continue
    
    # 解析XML文件
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    
    
    # 检查是否存在Pedestrian
    has_Pedestrian = False
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Pedestrian':
            has_Pedestrian = True
            break
    
    # 如果存在Pedestrian则复制文件
    if has_Pedestrian:
        base_name, ext = os.path.splitext(xml_file)
        for i in range(1, 2):
            new_xml_name = f"{base_name}_{i}.xml"
            new_img_name = f"{base_name}_{i}.jpg"
            new_json_name = f"{base_name}_{i}.json"
            new_seg_name = f"{base_name}_{i}.png"
            
            # 复制XML文件
            shutil.copyfile(os.path.join(xml_dir, xml_file), os.path.join(xml_dir, new_xml_name))
            
            # 复制图片文件
            shutil.copyfile(os.path.join(img_dir, base_name + '.jpg'), os.path.join(img_dir, new_img_name))
            processed_images_Pedestrian += 1
            
            # 复制JSON文件
            shutil.copyfile(os.path.join(json_dir, base_name + '.json'), os.path.join(json_dir, new_json_name))
            
            # 复制分割文件
            shutil.copyfile(os.path.join(seg_dir, base_name + '.png'), os.path.join(seg_dir, new_seg_name))
    
    # 检查是否存在Traffic Light-Yellow Light
    has_yellow_light = False
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Traffic Light-Yellow Light':
            has_yellow_light = True
            break
    
    # 如果存在Traffic Light-Yellow Light则复制文件
    if has_yellow_light:
        base_name, ext = os.path.splitext(xml_file)
        for i in range(1, 11):
            new_xml_name = f"{base_name}_{i}.xml"
            new_img_name = f"{base_name}_{i}.jpg"
            new_json_name = f"{base_name}_{i}.json"
            new_seg_name = f"{base_name}_{i}.png"
            
            # 复制XML文件
            shutil.copyfile(os.path.join(xml_dir, xml_file), os.path.join(xml_dir, new_xml_name))
            
            # 复制图片文件
            shutil.copyfile(os.path.join(img_dir, base_name + '.jpg'), os.path.join(img_dir, new_img_name))
            processed_images_yellow_light += 1
            
            # 复制JSON文件
            shutil.copyfile(os.path.join(json_dir, base_name + '.json'), os.path.join(json_dir, new_json_name))
            
            # 复制分割文件
            shutil.copyfile(os.path.join(seg_dir, base_name + '.png'), os.path.join(seg_dir, new_seg_name))
            
            
    # 检查是否存在Traffic off Light
    has_off_light = False
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Traffic Light-Off':
            has_off_light = True
            break
    
    # 如果存在Traffic off Light则复制文件
    if has_off_light:
        base_name, ext = os.path.splitext(xml_file)
        for i in range(1, 4):
            new_xml_name = f"{base_name}_{i}.xml"
            new_img_name = f"{base_name}_{i}.jpg"
            new_json_name = f"{base_name}_{i}.json"
            new_seg_name = f"{base_name}_{i}.png"
            
            # 复制XML文件
            shutil.copyfile(os.path.join(xml_dir, xml_file), os.path.join(xml_dir, new_xml_name))
            
            # 复制图片文件
            shutil.copyfile(os.path.join(img_dir, base_name + '.jpg'), os.path.join(img_dir, new_img_name))
            processed_images_off_light += 1
            
            # 复制JSON文件
            shutil.copyfile(os.path.join(json_dir, base_name + '.json'), os.path.join(json_dir, new_json_name))
            
            # 复制分割文件
            shutil.copyfile(os.path.join(seg_dir, base_name + '.png'), os.path.join(seg_dir, new_seg_name))
            
    # 检查是否存在Traffic red Light
    has_Light_Red_light = False
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Traffic Light-Red Light':
            has_Light_Red_light = True
            break
    
    # 如果存在Traffic off Light则复制文件
    if has_Light_Red_light:
        base_name, ext = os.path.splitext(xml_file)
        for i in range(1, 3):
            new_xml_name = f"{base_name}_{i}.xml"
            new_img_name = f"{base_name}_{i}.jpg"
            new_json_name = f"{base_name}_{i}.json"
            new_seg_name = f"{base_name}_{i}.png"
            
            # 复制XML文件
            shutil.copyfile(os.path.join(xml_dir, xml_file), os.path.join(xml_dir, new_xml_name))
            
            # 复制图片文件
            shutil.copyfile(os.path.join(img_dir, base_name + '.jpg'), os.path.join(img_dir, new_img_name))
            processed_images_Light_Red += 1
            
            # 复制JSON文件
            shutil.copyfile(os.path.join(json_dir, base_name + '.json'), os.path.join(json_dir, new_json_name))
            
            # 复制分割文件
            shutil.copyfile(os.path.join(seg_dir, base_name + '.png'), os.path.join(seg_dir, new_seg_name))
            
    
    # 检查是否存在Traffic green Light
    has_Light_green_light = False
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Traffic Light-Green Light':
            has_Light_green_light = True
            break
    
    # 如果存在Traffic off Light则复制文件
    if has_Light_green_light:
        base_name, ext = os.path.splitext(xml_file)
        for i in range(1, 3):
            new_xml_name = f"{base_name}_{i}.xml"
            new_img_name = f"{base_name}_{i}.jpg"
            new_json_name = f"{base_name}_{i}.json"
            new_seg_name = f"{base_name}_{i}.png"
            
            # 复制XML文件
            shutil.copyfile(os.path.join(xml_dir, xml_file), os.path.join(xml_dir, new_xml_name))
            
            # 复制图片文件
            shutil.copyfile(os.path.join(img_dir, base_name + '.jpg'), os.path.join(img_dir, new_img_name))
            processed_images_Light_green += 1
            
            # 复制JSON文件
            shutil.copyfile(os.path.join(json_dir, base_name + '.json'), os.path.join(json_dir, new_json_name))
            
            # 复制分割文件
            shutil.copyfile(os.path.join(seg_dir, base_name + '.png'), os.path.join(seg_dir, new_seg_name))
    
            
print(f"yellow-light: Processed {processed_images_yellow_light} images.")
print(f"off-light: Processed {processed_images_off_light} images.")
print(f"Light_Red: Processed {processed_images_Light_Red} images.")
print(f"Light_green: Processed {processed_images_Light_green} images.")
print(f"Pedestrian: Processed {processed_images_Pedestrian} images.")

