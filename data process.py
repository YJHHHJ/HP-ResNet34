import numpy as np
import os
import scipy.io.matlab as sio
import pywt

# 定义加载和处理数据的函数
def process_data(file_name, variable_name, save_folder):
    # 加载数据
    load_data = sio.loadmat(file_name)
    data = load_data[variable_name]  # 使用传入的变量名提取数据
    fs = 12000  # 采样频率，根据实际情况设定

    # 用于存储所有通道的小波系数矩阵的三维数组
    all_coefs = []

    # 对每个通道分别进行连续小波变换
    for channel in range(data.shape[1]):
        x = data[:, channel]
        wavename = 'cmor3-3'
        totalscal = 32
        Fc = pywt.central_frequency(wavename)
        c = 2 * Fc * totalscal
        scals = c / np.arange(1, totalscal + 1)
        f = pywt.scale2frequency(wavename, scals) * fs

        coefs, _ = pywt.cwt(x, scals, wavename, 1 / fs)
        all_coefs.append(coefs)
        print("每列数据系数形状:", coefs.shape)
    all_coefs = np.stack(all_coefs, axis=2)
    all_coefs = np.transpose(all_coefs, (1, 2, 0))
    print("合并后的系数形状:", all_coefs.shape)
    # 提取实部和虚部并重新组合成 (256, 40000, 6)
    real_parts = np.real(all_coefs)   # 实部，形状为 (256, 40000, 3)
    imag_parts = np.imag(all_coefs)   # 虚部，形状为 (256, 40000, 3)
    print("合并后的虚部形状:",imag_parts.shape)
    combined_coefs = np.concatenate((real_parts, imag_parts), axis=1)  # 合并为 (256, 40000, 6)
    print("合并后的全部形状:", combined_coefs.shape)
    # 划分训练集和验证集，这里取前30000个数据作为训练集，后30000个数据作为验证集

    train_data = combined_coefs[:30000, :, :]   # 修改为7000行
    val_data = combined_coefs[30000:, :, :]      # 后30000行

    # 切割训练集数据
    sample_length_train =32
    step_size_train = 2

    num_samples_to_extract_train = (train_data.shape[0] - sample_length_train) // step_size_train + 1

    sliced_train_data_list = []

    for i in range(num_samples_to_extract_train):
        start_idx = i * step_size_train
        end_idx = start_idx + sample_length_train

        if end_idx > train_data.shape[0]:  # 确保不超出范围
            break

        sliced_train_data_list.append(train_data[start_idx:end_idx].transpose(2, 0, 1))

    # 限制切片数量为80组，如果切片不足则取现有的切片数量
    sliced_train_data_list = sliced_train_data_list[:4000]

    final_train_data = np.stack(sliced_train_data_list).transpose(0, 3, 1, 2)

    # 保存训练集切片数据到指定路径
    train_save_path = os.path.join(save_folder, "train")
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)

    np.save(os.path.join(train_save_path, f"{os.path.basename(file_name).replace('.mat', '')}_train_sliced.npy"), final_train_data)

    print(f"{os.path.basename(file_name)}: 训练集的数据已成功切割并保存。")

    # 切割验证集数据
    num_samples_to_extract_val = (val_data.shape[0] - sample_length_train) // step_size_train + 1

    sliced_val_data_list = []

    for i in range(num_samples_to_extract_val):
        start_idx = i * step_size_train
        end_idx = start_idx + sample_length_train

        if end_idx > val_data.shape[0]:  # 确保不超出范围
            break

        sliced_val_data_list.append(val_data[start_idx:end_idx].transpose(2, 0, 1))

    # 限制切片数量为40组，如果切片不足则取现有的切片数量
    sliced_val_data_list = sliced_val_data_list[:1000]

    final_val_data = np.stack(sliced_val_data_list).transpose(0, 3, 1, 2)

    # 保存验证集切片数据到指定路径
    val_save_path = os.path.join(save_folder, "val")
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)

    np.save(os.path.join(val_save_path, f"{os.path.basename(file_name).replace('.mat', '')}_val_sliced.npy"), final_val_data)

    print(f"{os.path.basename(file_name)}: 验证集的数据已成功切割并保存。")

# 主程序部分：处理所有文件并合并结果
# file_names_and_vars = {
#    'y3097.mat': 'y097',
#    'y3105.mat': 'y105',
#    'y3118.mat': 'y118',
#    'y3130.mat': 'y130',
#    'y3169.mat': 'y169',
#    'y3185.mat': 'y185',
#    'y3197.mat': 'y197',
#    'y3209.mat': 'y209',
#    'y3222.mat': 'y222',
#    'y3234.mat': 'y234'
# }
file_names_and_vars = {
   'y_final100.mat': 'y_final',
   'y_final108.mat': 'y_final',
   'y_final121.mat': 'y_final',
   'y_final133.mat': 'y_final',
   'y_final172.mat': 'y_final',
   'y_final188.mat': 'y_final',
   'y_final200.mat': 'y_final',
   'y_final212.mat': 'y_final',
   'y_final225.mat': 'y_final',
   'y_final237.mat': 'y_final'
}


save_folder = "xichudax"

# 创建保存目录，如果不存在的话
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

final_combined_train_data_list = []
final_combined_val_data_list = []

for file_name, variable_name in file_names_and_vars.items():
    process_data(file_name, variable_name, save_folder)

# 加载所有训练和验证数据以进行合并
for file_name in file_names_and_vars.keys():
   train_file_path = os.path.join(save_folder,"train", f"{os.path.basename(file_name).replace('.mat', '')}_train_sliced.npy")
   val_file_path   = os.path.join(save_folder,"val", f"{os.path.basename(file_name).replace('.mat', '')}_val_sliced.npy")

   final_combined_train_data_list.append(np.load(train_file_path))
   final_combined_val_data_list.append(np.load(val_file_path))

# 合并所有训练和验证数据
final_combined_train_data = np.concatenate(final_combined_train_data_list)
final_combined_val_data   = np.concatenate(final_combined_val_data_list)

# 保存合并后的数据
np.save(os.path.join(save_folder,"train","combined_train_sliced.npy"), final_combined_train_data)
np.save(os.path.join(save_folder,"val","combined_val_sliced.npy"), final_combined_val_data)

print("所有数据已成功合并并保存。")
print("合并后的训练集形状:", final_combined_train_data.shape)
print("合并后的验证集形状:", final_combined_val_data.shape)
