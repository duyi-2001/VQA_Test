import os
import pandas as pd

# 读取 CSV 文件
csv_file = 'KoNViD_1k_attributes.csv'
df = pd.read_csv(csv_file)

# 设置视频文件所在的目录
video_directory = 'dataset/KoNViD_1k_videos'

# 遍历 CSV 文件中的每一行
for index, row in df.iterrows():
    original_name = str(row['flickr_id'])+'.mp4'
    new_name = row['file_name']

    # 检查原始文件是否存在
    original_file_path = os.path.join(video_directory, original_name)
    print(original_file_path)
    if os.path.isfile(original_file_path):
        # 构建新文件名的完整路径
        new_file_path = os.path.join(video_directory, new_name)

        # 重命名文件
        os.rename(original_file_path, new_file_path)
        print(f"Renamed {original_name} to {new_name}")
    else:
        print(f"Original file not found: {original_name}")
