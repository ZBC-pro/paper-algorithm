import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.cluster import KMeans
from python_speech_features import mfcc

# 预处理：对语音信号进行预加权，提升高频分量
def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    """
    对语音信号进行预加权。
    :param signal: 原始语音信号
    :param pre_emphasis_coeff: 预加权系数，通常为0.97
    :return: 预加权后的信号
    """

    return np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])

# MFCC特征提取
def extract_mfcc(signal, sample_rate, num_ceps=13, frame_size=0.025, frame_stride=0.01):
    """
    提取MFCC特征。
    :param signal: 输入语音信号
    :param sample_rate: 语音采样率
    :param num_ceps: 要提取的MFCC系数个数
    :param frame_size: 每帧的时间长度（秒）
    :param frame_stride: 帧移的时间长度（秒）10ms
    :return: 提取的MFCC特征
    """
    mfcc_features = mfcc(
        signal,
        samplerate=sample_rate,
        numcep=num_ceps,
        winlen=frame_size,
        winstep=frame_stride,
        preemph=0  # 加权系数为0，因为已经在预加权阶段完成
    )
    return mfcc_features

# 矢量量化：对特征进行k-means聚类
def vector_quantization(features, n_clusters=32):
    """
    使用k-means对特征进行矢量量化，生成码本。
    :param features: 输入特征
    :param n_clusters: 码本大小（聚类数）
    :return: 聚类模型和每个特征对应的码字索引
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    return kmeans, kmeans.predict(features)

# 遍历音频目录并提取特征
def process_directory(directory, n_clusters=32):
    """
    遍历目录中的音频文件，提取特征并生成码本。
    :param directory: 音频文件所在目录
    :param n_clusters: 聚类数（码本大小）
    :return: 提取的特征和标签
    """
    all_features = []
    labels = []
    for file_name in os.listdir(directory):
        print(file_name)
        if file_name in [i for i in range(0, 10)]:
            # 从文件名提取标签（数字部分）
            label = file_name
            for file in os.listdir(directory):
            # read 是 sci 函数，用于提取 wav 文件，sample_rate 是采样率，由音频文件质量决定，signal 是音频文件转化为离散数字信号的结果
            sample_rate, signal = read(os.path.join(directory, file_name))
            signal = pre_emphasis(signal)
            mfcc_features = extract_mfcc(signal, sample_rate)
            all_features.append(mfcc_features)
            labels.append(label)
    print(labels)


    # 矢量量化并生成码本
    flat_features = np.vstack(all_features)
    print(flat_features)
    kmeans_model, quantized_features = vector_quantization(flat_features, n_clusters)
    training_data = [(quantized_features[i], labels[i]) for i in range(len(labels))]

    return training_data, kmeans_model

if __name__ == "__main__":
    data_dir = r"/Users/dususu/Desktop/data/test/train/"  # 替换为数据集路径
    training_data, kmeans_model = process_directory(data_dir)

    # # 打印 K-Means 聚类结果
    # print("K-Means Cluster Centers:")
    # print(kmeans_model.cluster_centers_)  # 打印聚类中心
    #
    # print("\nCluster Assignment:")
    # # 打印每个数据点的分类结果
    # for i, label in enumerate(kmeans_model.labels_):
    #     print(f"Sample {i} assigned to Cluster {label}")