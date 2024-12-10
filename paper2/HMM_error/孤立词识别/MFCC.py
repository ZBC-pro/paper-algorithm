import os
import json
from scipy.io.wavfile import read
import numpy as np
from python_speech_features import mfcc  # 假设您使用的是 python_speech_features

def extract_mfcc(signal, sample_rate, num_ceps=13):
    """
    提取 MFCC 特征的函数。
    :param signal: 音频信号
    :param sample_rate: 采样率
    :param num_ceps: MFCC 特征维度
    :return: 提取的 MFCC 特征
    """
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=num_ceps)
    return mfcc_features

def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    """
    对语音信号进行预加权。
    :param signal: 输入信号
    :param pre_emphasis_coeff: 预加权系数
    :return: 预加权后的信号
    """
    return np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])

def process_dataset(directory):
    """
    遍历 0~9 文件夹，提取音频文件的 MFCC 特征并生成标签。
    :param directory: 根目录路径
    :return: 提取的特征和对应的标签
    """
    all_features = []
    labels = []

    # 遍历 0~9 的文件夹，确保文件夹按数字顺序排序
    for folder_name in sorted(os.listdir(directory), key=lambda x: int(x) if x.isdigit() else float('inf')):
        folder_path = os.path.join(directory, folder_name)

        # 检查文件夹名称是否为 0~9 的数字
        if os.path.isdir(folder_path) and folder_name.isdigit():
            label = int(folder_name)  # 文件夹名称作为标签

            # 遍历当前文件夹中的音频文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.endswith(".wav"):  # 只处理 .wav 文件
                    sample_rate, signal = read(file_path)  # 读取音频文件
                    signal = pre_emphasis(signal)  # 预加权处理
                    mfcc_features = extract_mfcc(signal, sample_rate)  # 提取 MFCC 特征
                    all_features.append(mfcc_features)  # 存储特征
                    labels.append(label)  # 存储对应的标签

    return all_features, labels

if __name__ == '__main__':
    dir_path = r"/Users/dususu/Desktop/data/test/train"
    all_features, labels = process_dataset(dir_path)

    # 创建一个字典存储每个标签对应的特征向量
    label_to_features = {}
    for feature, label in zip(all_features, labels):
        if label not in label_to_features:
            label_to_features[label] = []
        label_to_features[label].append(feature.tolist())  # 将 NumPy 数组转换为列表

    # 保存为 JSON 文件
    output_file = "label_features.json"
    with open(output_file, "w") as f:
        json.dump(label_to_features, f, indent=8)

    print(f"Features saved to {output_file}")