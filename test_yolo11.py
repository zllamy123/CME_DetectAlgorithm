from ultralytics import YOLO
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
######train################
#model = YOLO('yolo11-CBAM-cls.yaml').load("runs/classify/train31/weights/best.pt")  # load a pretrained model
model = YOLO("yolo11-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
results = model.train(data="/app/eyedetect/datasets", epochs=100, imgsz=224,classes= 3)
# metrics = model.val()

######testing################
# # 初始化模型
# model = YOLO("/app/ultralytics-main/runs/classify/train13/weights/best.pt")  # 替换为你的模型路径
# results = model("dataYuan/CNVYIDA/008.jpg")
# print(results[0].probs.top1)
# categories = ["CME", "NOCME"]  # 类别名称
# category_path = "dataYuan/非CMETest"
# for image_name in os.listdir(category_path):
#         image_path = os.path.join(category_path, image_name)
#         if os.path.isfile(image_path):  # 确保是文件
#             # 运行模型预测
#             results = model(image_path)
#             pred_label = results[0].probs.top1
#             if pred_label!=1:
#                 print("分类错误")
# #定义路径
# # test_path = "/app/eyedetect/datasets/test"  # 测试集根目录
# # categories = ["DME", "NODME"]  # 类别名称

# # # 收集真实标签和预测标签
# # y_true = []
# # y_pred = []

# # # 遍历测试集
# # for label, category in enumerate(categories):  # label: 0 for DME, 1 for NODME
# #     category_path = os.path.join(test_path, category)
# #     for image_name in os.listdir(category_path):
# #         image_path = os.path.join(category_path, image_name)
# #         if os.path.isfile(image_path):  # 确保是文件
# #             # 运行模型预测
# #             results = model(image_path)
          
# #             # # 显示预测结果（可选，保存图片结果）
# #             # results[0].show()  # 显示结果
# #             # save_path = os.path.join("results", category)  # 保存路径
# #             # os.makedirs(save_path, exist_ok=True)
# #             # results[
# # categories = ["CNV","DME", "Other"]  # 类别名称

# # # 收集真实标签和预测标签
# # y_true = []
# # y_pred = []

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

# # 计算混淆矩阵# ######################三分类指标##############################
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
# disp.plot(cmap="Blues", values_format="d")
# plt.title("Confusion Matrix")
# plt.show()



# values = cm.ravel()
# print(values) 
# # 提取TP、FP、FN、TN
# # Class 0
# TP_0 = values[0]
# FP_0 = values[1] + values[2]  # 假正例
# FN_0 = values[3] + values[6]  # 假负例
# TN_0 = values[4] + values[5] + values[7] + values[8]  # 真负例

# # Class 1
# TP_1 = values[4]
# FP_1 = values[3] + values[5]  # 假正例
# FN_1 = values[1] + values[7]  # 假负例
# TN_1 = values[0] + values[2] + values[6] + values[8]  # 真负例

# # Class 2
# TP_2 = values[8]
# FP_2 = values[6] + values[7]  # 假正例
# FN_2 = values[2] + values[5]  # 假负例
# TN_2 = values[0] + values[1] + values[3] + values[4]  # 真负例

# # 计算每个类别的灵敏度、特异度、精确率和 F1 分数
# def calculate_metrics(TP, FP, FN, TN):
#     # 灵敏度（召回率）
#     sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
#     # 特异度
#     specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
#     # 精确率
#     precision = TP / (TP + FP) if (TP + FP) != 0 else 0
#     # F1 分数
#     f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
#     return sensitivity, specificity, precision, f1

# # 计算每个类别的指标
# sensitivity_0, specificity_0, precision_0, f1_0 = calculate_metrics(TP_0, FP_0, FN_0, TN_0)
# sensitivity_1, specificity_1, precision_1, f1_1 = calculate_metrics(TP_1, FP_1, FN_1, TN_1)
# sensitivity_2, specificity_2, precision_2, f1_2 = calculate_metrics(TP_2, FP_2, FN_2, TN_2)

# # 计算总体准确率
# accuracy = (TP_0 + TP_1 + TP_2) / cm.sum()

# # 打印结果
# print(f"Class 0:")
# print(f"  灵敏度 (Sensitivity): {sensitivity_0:.2f}")
# print(f"  特异度 (Specificity): {specificity_0:.2f}")
# print(f"  精确率 (Precision): {precision_0:.2f}")
# print(f"  F1 分数: {f1_0:.2f}")
# print()

# print(f"Class 1:")
# print(f"  灵敏度 (Sensitivity): {sensitivity_1:.2f}")
# print(f"  特异度 (Specificity): {specificity_1:.2f}")
# print(f"  精确率 (Precision): {precision_1:.2f}")
# print(f"  F1 分数: {f1_1:.2f}")
# print()

# print(f"Class 2:")
# print(f"  灵敏度 (Sensitivity): {sensitivity_2:.2f}")
# print(f"  特异度 (Specificity): {specificity_2:.2f}")
# print(f"  精确率 (Precision): {precision_2:.2f}")
# print(f"  F1 分数: {f1_2:.2f}")
# print()

# # 总体准确率
# print(f"总体准确率 (Accuracy): {accuracy:.2f}")
# ######################三分类指标##############################



######################二分类指标##############################

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