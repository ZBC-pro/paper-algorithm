'''
用维特比算法求解最可能的初始状态转移矩阵
'''
import json
import numpy as np
from hmmlearn.hmm import CategoricalHMM


def load_observation_sequences(json_file):
    """
    从 JSON 文件加载观测序列。
    :param json_file: JSON 文件路径。
    :return: 包含每个标签观测序列的字典。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def create_hmm_model(n_states, n_observations):
    """
    创建一个 HMM 模型。
    :param n_states: 隐藏状态的数量。
    :param n_observations: 观测状态的数量。
    :return: 初始化的 HMM 模型。
    """
    model = CategoricalHMM(n_components=n_states, random_state=42, n_iter=100, tol=1e-4, init_params="")

    # 随机初始化参数
    model.startprob_ = np.full(n_states, 1 / n_states)  # 平均分布的初始概率
    model.transmat_ = np.full((n_states, n_states), 1 / n_states)  # 平均分布的状态转移矩阵
    model.emissionprob_ = np.full((n_states, n_observations), 1 / n_observations)  # 平均分布的发射概率矩阵

    return model


def compute_hidden_states(model, observation_sequence):
    """
    使用维特比算法计算最可能的隐藏状态序列。
    :param model: 已初始化的 HMM 模型。
    :param observation_sequence: 观测序列 (1D numpy 数组)。
    :return: 最可能的隐藏状态序列 (1D numpy 数组)。
    """
    logprob, hidden_states = model.decode(np.array(observation_sequence).reshape(-1, 1), algorithm="viterbi")
    return logprob, hidden_states


if __name__ == "__main__":
    # 加载 JSON 文件中的观测序列
    json_file = r"labeled_observation_sequences.json"  # 替换为你的文件路径
    observation_sequences = load_observation_sequences(json_file)

    # 设置 HMM 参数
    n_hidden_states = 4  # 隐藏状态的数量
    n_observations = 12  # 观测状态的数量

    # 初始化 HMM 模型
    hmm_model = create_hmm_model(n_hidden_states, n_observations)

    # 遍历每个标签的观测序列
    for label, sequence in observation_sequences.items():
        print(f"\n标签 {label} 的观测序列: {sequence}")

        # 拟合 HMM 模型（用单个序列训练）
        sequence_array = np.array(sequence).reshape(-1, 1)
        hmm_model.fit(sequence_array)

        # 使用维特比算法计算最可能的隐藏状态序列
        logprob, hidden_states = compute_hidden_states(hmm_model, sequence)
        print(f"标签 {label} 的最可能隐藏状态序列: {hidden_states}")
        print(f"对数概率: {logprob}")