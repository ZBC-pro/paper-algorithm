'''
json输出格式是label + 特征向量
'''

import os
import json
import librosa
import numpy as np
from python_speech_features import mfcc


def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    """
    从音频文件中提取 MFCC 特征。
    :param file_path: 音频文件路径。
    :param sr: 采样率。
    :param n_mfcc: MFCC 特征维数。
    :return: MFCC 特征矩阵。
    """
    y, _ = librosa.load(file_path, sr=sr)
    mfcc_features = mfcc(y, samplerate=sr, numcep=n_mfcc)
    return mfcc_features.tolist()  # 转换为列表以便存储为 JSON 格式


def process_folder(folder_path, label, sr=16000, n_mfcc=13):
    """
    处理指定文件夹内的所有音频文件，并提取 MFCC 特征。
    :param folder_path: 文件夹路径。
    :param label: 文件夹对应的标签。
    :param sr: 采样率。
    :param n_mfcc: MFCC 特征维数。
    :return: 一个包含标签和对应特征的列表。
    """
    feature_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.wav'):  # 确保处理的是 WAV 文件
            print(f"Processing file: {file_path}")
            mfcc_features = extract_mfcc(file_path, sr=sr, n_mfcc=n_mfcc)
            feature_list.append({"label": label, "features": mfcc_features})
    return feature_list


def main(input_dir, output_json):
    """
    主函数：遍历 0～9 的文件夹，提取所有特征并存储为 JSON 文件。
    :param input_dir: 包含 0～9 文件夹的根目录。
    :param output_json: 输出 JSON 文件路径。
    """
    all_features = []

    for label in range(10):
        folder_path = os.path.join(input_dir, str(label))  # 文件夹路径
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path} with label {label}")
            folder_features = process_folder(folder_path, label)
            all_features.extend(folder_features)

    # 将结果保存为 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(all_features, f, indent=4)
    print(f"All features have been saved to {output_json}")


if __name__ == "__main__":
    # 输入文件夹路径和输出 JSON 文件路径
    input_directory = r"/Users/dususu/Desktop/data/test/train"
    output_file = "labeled_mfcc_features.json"  # 输出 JSON 文件的路径

    # 运行主程序
    main(input_directory, output_file)