from ultralytics import YOLO
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay


import matplotlib.pyplot as plt
######train################
#model = YOLO('yolo11-CBAM-cls.yaml').load("runs/classify/train31/weights/best.pt")  # load a pretrained model
model = YOLO("yolo11-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
results = model.train(data="/app/eyedetect/datasets", epochs=1, imgsz=224,classes= 3)
# metrics = model.val()
######testing################
# # 初始化模型
model = YOLO("/app/ultralytics-main/runs/classify/train13/weights/best.pt")  # 替换为你的模型路径
# model.export(format="onnx")  # 导出为 ONNX 格式
# category_path = r"test_data_打针2/2次后"
# output_file_path = "test_data_打针2/results/2针后.txt"  # 预测结果保存的TXT文件路径
category_path = r"test_data_打针2/test"
output_file_path = "test_data_打针2/results/test.txt"  # 预测结果保存的TXT文件路径
categories = ["CME", "NOCME"]  # 类别名称

with open(output_file_path, 'w') as output_file:
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        if os.path.isfile(image_path):  # 确保是文件
            # 运行模型预测
            results = model(image_path)
            prob = results[0].probs.top1conf  # 获取最高概率值
            pred_label = results[0].probs.top1  # 获取预测标签
          
           

            # 写入文件名和预测标签
            output_file.write(f"{image_name}: {categories[pred_label]}:{prob }\n")







# #####testing################
# # 初始化模型
# model = YOLO("/app/ultralytics-main/runs/classify/train13/weights/best.pt")  # 替换为你的模型路径
# test_path = "/app/eyedetect/datasets/test"  # 测试集根目录
# categories = ["CME", "NOCME"]  # 类别名称

# # 收集真实标签和预测标签
# y_true = []
# y_pred = []

# # 遍历测试集
# for label, category in enumerate(categories):  # label: 0 for DME, 1 for NODME
#     category_path = os.path.join(test_path, category)
#     for image_name in os.listdir(category_path):
#         image_path = os.path.join(category_path, image_name)
#         if os.path.isfile(image_path):  # 确保是文件
#             # 运行模型预测
#             results = model(image_path)
          
#             # # 显示预测结果（可选，保存图片结果）
#             # results[0].show()  # 显示结果
#             # save_path = os.path.join("results", category)  # 保存路径
#             # os.makedirs(save_path, exist_ok=True)
#             # results[0].save(save_path)  # 保存预测结果

#             # 获取预测标签（假设结果为二分类概率）
#             pred_label = results[0].probs.top1
            
#             # 记录真实标签和预测标签
#             y_true.append(label)
#             y_pred.append(pred_label)


# ######################二分类指标##############################

# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
# disp.plot(cmap="Blues", values_format="d")
# plt.title("Confusion Matrix")
# plt.show()
# tn, fp, fn, tp = cm.ravel()

# # 计算指标
# accuracy = accuracy_score(y_true, y_pred)
# sensitivity = tp / (tp + fn)  # 灵敏度
# specificity = tn / (tn + fp)  # 特异度
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率
# f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# # 输出结果
# print("混淆矩阵:")
# print(cm)
# print(f"准确率 (Accuracy): {accuracy:.2f}")
# print(f"灵敏度 (Sensitivity/Recall): {sensitivity:.2f}")
# print(f"特异度 (Specificity): {specificity:.2f}")
# print(f"精确率 (Precision): {precision:.2f}")
# print(f"F1 分数: {f1_score:.2f}")

# # 分类报告
# print("\n分类报告:")
# print(classification_report(y_true, y_pred, target_names=categories))


