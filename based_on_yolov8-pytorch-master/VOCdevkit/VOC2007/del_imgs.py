import os

# 设置文件夹路径，将这里的'your_directory_path'替换为您存放.xml文件的实际路径
your_directory_path = './JPEGImages'

# 列出文件夹中所有文件
files_in_directory = os.listdir(your_directory_path)

# 过滤出文件名中含有下划线的.xml文件
files_with_underscore = [f for f in files_in_directory if '_' in f and f.endswith('.jpg')]

# 删除这些文件
for file in files_with_underscore:
    os.remove(os.path.join(your_directory_path, file))
    print(f"Deleted file: {file}")  # 打印出被删除的文件名

# 如果需要确认文件是否被删除，可以再次列出目录中的文件并打印
remaining_files = os.listdir(your_directory_path)
#print("Remaining files:")
#print(remaining_files)
