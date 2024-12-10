'''
读取json中的label的第一条特征向量，作为k-means的参照，以求解HMM的初始参数
'''
import json
import numpy as np
from sklearn.cluster import KMeans


def read_first_features(json_file):
    """
    读取 JSON 文件中每个标签的第一条特征数据。
    :param json_file: JSON 文件路径。
    :return: 包含每个标签的第一条特征数据的字典。
    """
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 初始化一个字典，用于存储每个标签的第一条特征数据
    first_features = {}

    # 遍历数据，提取每个标签的第一条特征
    for entry in data:
        label = entry['label']  # 获取标签
        if label not in first_features:  # 确保只取第一条特征
            first_features[label] = entry['features']

    return first_features


def perform_kmeans_per_label(first_features, n_clusters):
    """
    对每个标签的特征向量单独执行 K-Means 聚类，生成观测序列。
    :param first_features: 包含每个标签的第一条特征数据的字典。
    :param n_clusters: K-Means 的簇数量。
    :return: 每个标签对应的观测序列字典。
    """
    observation_sequences = {}

    for label, features in first_features.items():
        features = np.array(features)  # 转为 numpy 数组
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features)  # 聚类
        observation_sequences[label] = kmeans_labels.tolist()  # 保存观测序列为列表
        print(f"标签 {label} 的观测序列: {kmeans_labels.tolist()}")  # 打印结果

    return observation_sequences


def save_observation_sequences_to_json(observation_sequences, output_file):
    """
    将观测序列保存为 JSON 文件。
    :param observation_sequences: 每个标签对应的观测序列字典。
    :param output_file: 输出 JSON 文件路径。
    """
    # 确保所有数据是 Python 原生类型
    print(observation_sequences)
    for label in observation_sequences:
        observation_sequences[label] = observation_sequences[label]

    # 写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(observation_sequences, f, indent=4)


if __name__ == "__main__":
    json_file = r"../labeled_mfcc_features.json"
    output_file = r"labeled_observation_sequences.json"

    # 读取第一条特征向量
    first_features = read_first_features(json_file)
    for label, features in first_features.items():
        print(f"标签: {label}, 特征长度:", len(features))
        print(f"Label: {label}, First Feature: {features[:5]}...", "\n")  # 仅打印前 5 个特征数据

    # 设置 HMM 参数
    n_hidden_states = 5  # HMM 的隐藏状态数量
    n_observations = 10  # HMM 的观测值数量

    # 对每个标签单独执行 K-Means 聚类，生成观测序列
    observation_sequences = perform_kmeans_per_label(first_features, n_observations)

    # 保存观测序列至 JSON 文件
    save_observation_sequences_to_json(observation_sequences, output_file)