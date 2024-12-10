import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colormaps


# 限制发射矩阵的大小
def enforce_minimum_probability(B, min_value=0.025):
    """
    限制发射矩阵的最小值，并保持每行归一化。
    :param B: 发射矩阵
    :param min_value: 发射矩阵中允许的最小值
    :return: 调整后的发射矩阵
    """
    B = np.maximum(B, min_value)  # 将矩阵中的值限制在 min_value 以上
    row_sums = B.sum(axis=1, keepdims=True)  # 计算每行的和
    B = B / row_sums  # 归一化
    return B


# 维特比算法--确定初始隐藏状态矩阵
def viterbi(pi, A, B, observations):
    """
    使用维特比算法计算最优隐藏状态序列。
    :param pi: 初始状态概率分布 (n_states,)
    :param A: 状态转移矩阵 (n_states, n_states)
    :param B: 观测概率矩阵 (n_states, n_symbols)
    :param observations: 观测序列 (list of int)
    :return: 最优隐藏状态序列
    """
    n_states = A.shape[0]
    T = len(observations)

    # 初始化动态规划表
    dp = np.zeros((T, n_states))  # 存储每个时间步的最大概率
    backpointer = np.zeros((T, n_states), dtype=int)  # 回溯路径

    # 初始化第一步
    dp[0, :] = pi * B[:, observations[0]]

    # 动态规划填表
    for t in range(1, T):
        for j in range(n_states):
            probabilities = dp[t - 1, :] * A[:, j] * B[j, observations[t]]
            dp[t, j] = np.max(probabilities)
            backpointer[t, j] = np.argmax(probabilities)

    # 回溯路径
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(dp[-1, :])  # 找到最后一步的最优状态
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path


# 确定HMM参数
def initialize_discrete_hmm(features, n_states, n_symbols):
    """
    使用离散方法初始化 HMM 参数。
    :param features: 一个标签下的所有特征向量 (list of ndarray)
    :param n_states: 隐状态数量
    :param n_symbols: 离散观测符号数量
    :return: 初始状态概率 pi, 状态转移矩阵 A, 观测概率矩阵 B, symbols
    """
    # 将所有特征向量合并为一个数组
    all_features = np.vstack(features)
    print(f"Feature array shape: {all_features.shape}")

    # Step 1: 使用 KMeans 聚类生成离散符号
    kmeans = KMeans(n_clusters=n_symbols, random_state=42)
    symbols = kmeans.fit_predict(all_features)  # 将特征向量映射为离散符号
    print(f"Generated observation sequence (symbols): {symbols}")  # 打印生成的观测序列
    print("观测序列长度：",len(symbols))

    # 可视化 K-means 聚类结果
    visualize_kmeans(all_features, symbols, n_symbols)

    # 将每个特征序列转化为离散观测符号序列
    sequences = []
    start = 0
    for feature in features:
        length = feature.shape[0]
        sequences.append(symbols[start:start + length])
        start += length

    # Step 2: 初始化 HMM 参数
    # 初始化 π（初始状态分布）
    pi = np.zeros(n_states)
    for seq in sequences:
        pi[seq[0] % n_states] += 1  # 假设状态与符号有初步对应关系
    pi = pi / pi.sum()  # 归一化

    # 初始化 A（状态转移矩阵）
    A = np.zeros((n_states, n_states))
    for seq in sequences:
        for i in range(len(seq) - 1):
            A[seq[i] % n_states, seq[i + 1] % n_states] += 1
    A = A / A.sum(axis=1, keepdims=True)  # 归一化

    # 初始化 B（观测概率矩阵）
    B = np.zeros((n_states, n_symbols))
    for seq in sequences:
        for obs in seq:
            B[obs % n_states, obs] += 1
    B = B / B.sum(axis=1, keepdims=True)  # 归一化

    # 对发射矩阵 B 进行限制
    B = enforce_minimum_probability(B)

    return pi, A, B, symbols


def visualize_kmeans(features, labels, n_clusters):
    """
    可视化 K-means 聚类结果（降维到 2D）。
    :param features: 原始特征数据 (ndarray)
    :param labels: 聚类结果标签 (ndarray)
    :param n_clusters: 聚类数量
    """
    # 使用 PCA 将高维数据降到二维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # 创建一个颜色列表用于区分聚类
    colors = colormaps["tab10"]

    # 绘制每个点，颜色根据聚类标签分配
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_points = reduced_features[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", color=colors(cluster))

    plt.title("K-means Clustering Visualization")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.grid()
    # plt.show()


# 加载json文件内容，并且转化为可合并的np数组，后续将所有np数组连接起来，拼接后对所有label对应的特征向量进行k-means聚类，找出观测序列
def load_features_from_json(file_path):
    """
    从 JSON 文件加载特征数据。
    :param file_path: JSON 文件路径
    :return: 一个包含标签及对应特征的字典
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 将特征从列表转化为 numpy 数组
    label_to_features = {int(label): [np.array(features) for features in feature_list]
                         for label, feature_list in data.items()}

    features_label_0 = label_to_features[0]  # 这是一个 list，包含多个 np.array
    shape_of_first_feature = features_label_0[0].shape  # 取出第一个特征，调用 .shape
    print("0 的特征形状",shape_of_first_feature)
    print("***")

    return label_to_features


# 保存初始值到json文件中
def save_hmm_parameters_to_json(output_path, hmm_parameters):
    """
    将 HMM 参数保存为 JSON 文件。
    :param output_path: 输出 JSON 文件路径
    :param hmm_parameters: HMM 参数字典 {label: {"pi": ..., "A": ..., "B": ...}}
    """
    # 转换 numpy 数组为列表以便 JSON 序列化
    hmm_parameters_serializable = {
        label: {
            "pi": params["pi"].tolist(),
            "A": params["A"].tolist(),
            "B": params["B"].tolist()
        }
        for label, params in hmm_parameters.items()
    }
    with open(output_path, 'w') as f:
        json.dump(hmm_parameters_serializable, f, indent=4)
    print(f"HMM parameters saved to {output_path}")



if __name__ == '__main__':
    # 加载特征数据
    json_path = "label_features.json"  # 替换为实际路径
    label_to_features = load_features_from_json(json_path)

    # HMM 初始化
    n_states = 5  # 假设每个 HMM 使用 5 个隐状态
    n_symbols = 10  # 假设离散观测符号数量为 10

    # 初始化一个字典存储所有标签的 HMM 参数
    hmm_parameters = {}

    # 遍历每个标签，计算对应的 HMM 参数
    for label, features in label_to_features.items():
        pi, A, B, symbols = initialize_discrete_hmm(features, n_states, n_symbols)
        print(f"Label {label}:")
        print("Generated observation sequence (symbols):", symbols)  # 输出观测序列
        print("Initial State Distribution (π):", pi)
        print("State Transition Matrix (A):", A)
        print("Observation Probability Matrix (B):", B)
        print('-' * 80)

        # 使用维特比算法找出每个序列的最优隐藏状态路径
        sequences = []
        start = 0
        for feature in features:
            length = feature.shape[0]
            sequences.append(symbols[start:start + length])
            start += length

        # 找到所有序列的隐藏状态路径
        optimal_hidden_states = []
        for observation in sequences:
            hidden_states = viterbi(pi, A, B, observation)
            optimal_hidden_states.append(hidden_states)
            print(f"Optimal hidden state sequence for label {label}: {hidden_states}")

        # 根据最优隐藏状态路径重新计算状态转移矩阵 A
        new_A = np.zeros((n_states, n_states))
        for hidden_states in optimal_hidden_states:
            for t in range(len(hidden_states) - 1):
                new_A[hidden_states[t], hidden_states[t + 1]] += 1
        new_A = new_A / new_A.sum(axis=1, keepdims=True)  # 归一化

        print(f"Updated State Transition Matrix (A) for label {label}: {new_A}")
        print('-' * 80)

        # 保存该标签的 HMM 参数
        hmm_parameters[label] = {"pi": pi, "A": new_A, "B": B}

    # 将所有 HMM 参数保存到 JSON 文件
    output_path = "hmm_parameters.json"  # 指定保存路径
    save_hmm_parameters_to_json(output_path, hmm_parameters)