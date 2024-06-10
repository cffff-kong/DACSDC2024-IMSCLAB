import os
import json
from PIL import Image, ImageDraw
from xml.etree import ElementTree as ET

def process_files(xml_folder, json_folder, output_folder):
    # 获取所有的XML文件
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]

    for xml_file in xml_files:
        # 构建JSON文件的路径
        json_file = xml_file.replace('.xml', '.json')

        # 检查JSON文件是否存在
        if not os.path.exists(os.path.join(json_folder, json_file)):
            print(f"JSON file for {xml_file} not found.")
            continue

        # 解析XML文件
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        size_element = root.find('size')
        width = int(size_element.find('width').text)
        height = int(size_element.find('height').text)

        # 加载JSON数据
        with open(os.path.join(json_folder, json_file), 'r') as file:
            json_data = json.load(file)

        # 创建图像
        image = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(image)

        # 绘制分割区域
        for obj in json_data:
            if obj['type'] in [8, 9, 10] and obj['segmentation']:
                segmentation = obj['segmentation'][0]
                points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                draw.polygon(points, fill=obj['type'])

        # 保存图像
        output_image_path = os.path.join(output_folder, xml_file.replace('.xml', '.png'))
        image.save(output_image_path)

# 设置文件夹路径
xml_folder = './Annotations'
json_folder = './label'
output_folder = './seg'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理所有文件
process_files(xml_folder, json_folder, output_folder)
