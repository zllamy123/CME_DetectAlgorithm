import os
from PIL import Image
from ultralytics import YOLO
import torch

# 中央裁剪函数
def central_crop(image, target_size):
    width, height = image.size
    crop_width, crop_height = target_size
    
    # 计算裁剪区域的位置
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2
    
    # 裁剪图片
    return image.crop((left, top, right, bottom))

# 加载YOLO模型
model = YOLO("/app/ultralytics-main/runs/classify/train10/weights/best.pt")  # 替换为你的模型路径

category_path = "dataYuan/CME处理后"
target_size = (512, 512)  # 你想裁剪的尺寸（例如 512x512）
save_dir = "dataYuan/CMEcaijian"  # 用于保存裁剪后图片的目录

# 创建保存图片的文件夹（如果不存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for image_name in os.listdir(category_path):
    image_path = os.path.join(category_path, image_name)
    if os.path.isfile(image_path):  # 确保是文件
        try:
            # 打开图片并进行中央裁剪
            with Image.open(image_path) as img:
                cropped_img = central_crop(img, target_size)
                
                # 保存裁剪后的图片
                cropped_image_path = os.path.join(save_dir, f"cropped_{image_name}")
                cropped_img.save(cropped_image_path)
                
                # 将裁剪后的图片传入模型进行预测
                results = model(cropped_img)
                
                # 获取预测标签
                pred_label = results[0].probs.top1
                if pred_label != 1:
                    print(f"{image_name} 分类错误")
                else:
                    print(f"Predicted label for {image_name}: {pred_label}")
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
