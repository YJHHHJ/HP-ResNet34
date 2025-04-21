"""
Created on Wed Aug  5 14:09:35 2020
@author: 17505
一维识别
"""
from ctypes import pydll
import torch
import time
from io import StringIO
import scipy.io as sio
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
from g8 import ResNet34

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet34().to(device)

start_time = time.time()
EPOCH = 100
BATCH_SIZE = 100
LR = 0.0000001

### 数据读取 ###
data_root_folder = "xichudax"

# 训练集数据加载
train_folder_path = os.path.join(data_root_folder, "train")
train_file_path = os.path.join(train_folder_path, "combined_train_sliced.npy")
if os.path.exists(train_file_path):
    _data1= np.load(train_file_path)
else:
    print(f"训练集数据文件 {train_file_path} 不存在，请检查路径和文件是否正确。")

# 验证集数据加载
val_folder_path = os.path.join(data_root_folder, "val")
val_file_path = os.path.join(val_folder_path, "combined_val_sliced.npy")
if os.path.exists(val_file_path):
    _data_val1 = np.load(val_file_path)
else:
    print(f"验证集数据文件 {val_file_path} 不存在，请检查路径和文件是否正确。")

def normalization(data):
    _range = np.max(abs(data))
    return data / _range
_data=normalization(_data1)
_data_val=normalization(_data_val1)

# 为训练集打标签
y1_data = np.zeros(40000)
y1_data[4000:8000] = 1
y1_data[8000:12000] = 2
y1_data[12000:16000] = 3
y1_data[16000:20000] = 4
y1_data[20000:24000] = 5
y1_data[24000:28000] = 6
y1_data[28000:32000] = 7
y1_data[32000:36000] = 8
y1_data[36000:40000] = 9

os.makedirs('labels_data', exist_ok=True)
np.save(os.path.join('labels_data', 'y1_data.npy'), y1_data)
x1_data = torch.from_numpy(_data).float()
y1_data = torch.from_numpy(y1_data).float()

# 为验证集打标签
y2_data=np.zeros(10000)
y2_data[1000:2000]=1
y2_data[2000:3000]=2
y2_data[3000:4000]=3
y2_data[4000:5000]=4
y2_data[5000:6000]=5
y2_data[6000:7000]=6
y2_data[7000:8000]=7
y2_data[8000:9000]=8
y2_data[9000:10000] = 9

os.makedirs('labels_data', exist_ok=True)
np.save(os.path.join('labels_data', 'y2_data.npy'), y2_data)
x2_data = torch.from_numpy(_data_val).float()
y2_data = torch.from_numpy(y2_data).float()

print("训练集大小:", len(x1_data))
print("验证集大小:", len(x2_data))

deal_dataset1 = TensorDataset(x1_data, y1_data)
trainloader = DataLoader(dataset=deal_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

deal_dataset2 = TensorDataset(x2_data, y2_data)
testloader = DataLoader(dataset=deal_dataset2, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

L0 = len(x1_data[1, :])
L1 = len(trainloader)
L2 = len(testloader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

sLoss_list = []
vLoss_list = []
sCorrect_list = []
vCorrect_list = []
best_correct = 0
save_path = './net.pth'

print("Start Training")
for epoch in range(EPOCH):
    if epoch % 10 == 0:
        LR *= 0.95

    net.train()
    s_loss = 0.0
    v_loss = 0.0
    s_correct = 0.
    s_total = 0.

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        s_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        s_total += labels.size(0)
        s_correct += predicted.eq(labels.data).cpu().sum()

        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * len(trainloader)), s_loss / (i + 1), 100. * s_correct / s_total))

    sCorrect_list.append(100 * (s_correct / len(deal_dataset1)).item())

    print("Waiting Test!")
    with torch.no_grad():
        v_correct = 0
        v_total = 0
        all_labels = []
        all_predictions = []

        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device=device, dtype=torch.int64)
            outputs = net(images)
            val_loss = criterion(outputs, labels)
            v_loss += val_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            v_total += labels.size(0)
            v_correct += (predicted == labels).sum()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 只在最后一个epoch绘制并保存混淆矩阵
            if epoch == EPOCH - 1:
                cm = confusion_matrix(all_labels, all_predictions, labels=list(range(10)))

                plt.figure(figsize=(7 / 2.54, 6 / 2.54))  # 保持7x6cm尺寸不变
                sns.set(font='Times New Roman')
                ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                 xticklabels=[f"{i}" for i in range(10)],
                                 yticklabels=[f"{i}" for i in range(10)],
                                 annot_kws={
                                     "size": 6,
                                     'fontfamily': 'Times New Roman'
                                 },
                                 cbar_kws={
                                     "shrink": 0.7,
                                     "aspect": 10,  # 调整颜色条长宽比
                                     "drawedges": False  # 移除边框线
                                 })

                # 获取颜色条对象并设置字体
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=8)  # 设置颜色条刻度字体为8pt

                ax.set_title(f'(d1) Confusion Matrix\nVal Acc: {100 * v_correct / v_total:.2f}%',
                             fontsize=8, pad=10, fontfamily='Times New Roman')
                ax.set_xlabel('Predicted', fontsize=8, fontfamily='Times New Roman')
                ax.set_ylabel('True', fontsize=8, fontfamily='Times New Roman')
                ax.tick_params(labelsize=7)  # 保持坐标轴标签为7pt
            os.makedirs('E:\论文\完整实验\ResNet34创新', exist_ok=True)

            # 保存图像
            cm_filename = 'E:\论文\完整实验\ResNet34创新/final_confusion_matrix.svg'
            plt.savefig(cm_filename, bbox_inches='tight', dpi=600, format='svg')
            plt.close()

        vCorrect_list.append(100 * (v_correct / len(deal_dataset2)).item())
        print('测试分类准确率为：%.3f%%' % (100 * (s_correct / s_total).item()))
        print('验证分类准确率为：%.3f%%' % (100 * (v_correct / v_total).item()))

    if v_correct > best_correct:
        best_correct = v_correct
        torch.save(net.state_dict(), save_path)

    sLoss_list.append(s_loss / L1)
    vLoss_list.append(v_loss / L2)

print('finished training')
end_time = time.time()
print(f"运行时间：{end_time - start_time:.6f}秒")

### 输出最后一次的训练集和验证集损失值 ###
print("\n训练结束，最终损失值：")
print(f"训练集损失: {sLoss_list[-1]:.6f}")
print(f"验证集损失: {vLoss_list[-1]:.6f}")

### 绘图 ###
# 在绘图部分修改为以下代码（只替换绘图部分，其他代码保持不变）

### 绘图 ###
x_range = range(1, EPOCH + 1)

# 设置全局绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 600,
})

# 创建7x5cm的图形
plt.figure(figsize=(8/2.54,7/2.54))

x_range = range(1, EPOCH + 1)

# 设置全局绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 600,
})

# 创建7x5cm的图形
plt.figure(figsize=(8/2.54,7/2.54))

# 第一个子图：损失曲线（训练集蓝色，验证集红色）
plt.subplot(2, 1, 1)
plt.plot(x_range, sLoss_list, 'b.-', linewidth=0.5, markersize=5, label='Train Loss')
plt.plot(x_range, vLoss_list, 'r.-', linewidth=0.5, markersize=5, label='Validation Loss')
plt.title('(d1) Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 将图例放在右上角，并设置边框和背景透明
plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)

# 第二个子图：准确率曲线（训练集蓝色，验证集红色）
plt.subplot(2, 1, 2)
plt.plot(x_range, sCorrect_list, 'bo-', linewidth=0.5, markersize=3, label='Train Accuracy')
plt.plot(x_range, vCorrect_list, 'ro-', linewidth=0.5, markersize=3, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
# 将图例放在右下角，并设置边框和背景透明
plt.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.0, 0.0), borderaxespad=0.)


# 确保目录存在
os.makedirs('E:\论文\完整实验\ResNet34创新', exist_ok=True)

# 保存图像（使用绝对路径更可靠）
metrics_filename = os.path.abspath('E:\论文\完整实验\ResNet34创新/training_metrics.svg')
plt.savefig(metrics_filename, bbox_inches='tight', dpi=600, format='svg')
plt.close()  # 确保图形被正确关闭
print(f"Saved training metrics to {metrics_filename}")

target_folder = "E:\论文\完整实验\ResNet34创新"  # 相对路径
os.makedirs(target_folder, exist_ok=True)
# 保存文件
sio.savemat(os.path.join(target_folder, 'loss_train.mat'), {'loss_train': np.array(sLoss_list)})
sio.savemat(os.path.join(target_folder, 'loss_validation.mat'), {'loss_validation': np.array(vLoss_list)})
sio.savemat(os.path.join(target_folder, 'accuracy_train.mat'), {'accuracy_train': np.array(sCorrect_list)})
sio.savemat(os.path.join(target_folder, 'accuracy_validation.mat'), {'accuracy_validation': np.array(vCorrect_list)})

print(f"文件已保存到：{os.path.abspath(target_folder)}")