import os
import shutil
import random

# 原始数据集路径
source_folder = 'dataYuan_2 _copy'  # 数据集文件夹，包含DME和NODME子文件夹

# 目标文件夹路径
train_folder = 'datasets/train'
val_folder = 'datasets/val'
test_folder = 'datasets/test'

# 创建目标文件夹
for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 为每个子类别（DME, NODME）创建对应的文件夹
for folder in [train_folder, val_folder, test_folder]:
    for category in ["CNV", "DME", "Other"]:
        category_folder = os.path.join(folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

# 定义数据分配比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 获取源文件夹中的所有类别（DME, NODME）
for category in ["CNV", "DME", "Other"]:
    source_category_folder = os.path.join(source_folder, category)
    if not os.path.exists(source_category_folder):
        print(f"源文件夹 {source_category_folder} 不存在!")
        continue

    # 获取该类别下的所有文件
    files = os.listdir(source_category_folder)
    
    # 随机打乱文件列表
    random.shuffle(files)

    # 计算各子集的大小
    total_files = len(files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    # 划分文件
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # 定义目标文件夹
    train_category_folder = os.path.join(train_folder, category)
    val_category_folder = os.path.join(val_folder, category)
    test_category_folder = os.path.join(test_folder, category)

    # 将文件移动到相应的文件夹
    for file in train_files:
        shutil.move(os.path.join(source_category_folder, file), os.path.join(train_category_folder, file))

    for file in val_files:
        shutil.move(os.path.join(source_category_folder, file), os.path.join(val_category_folder, file))

    for file in test_files:
        shutil.move(os.path.join(source_category_folder, file), os.path.join(test_category_folder, file))

    print(f"{category} 类别处理完成：训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张, 测试集 {len(test_files)} 张")
