import numpy as np
from hmmlearn.hmm import MultinomialHMM


def create_hmm_model(pi, A, B):
    """
    创建并初始化一个 MultinomialHMM 模型。

    :param pi: 初始状态概率分布 (1D numpy 数组)。
    :param A: 状态转移概率矩阵 (2D numpy 数组)。
    :param B: 发射概率矩阵 (2D numpy 数组)。
    :return: 初始化后的 MultinomialHMM 模型。
    """
    n_states = len(pi)  # 隐藏状态的数量
    n_symbols = B.shape[1]  # 观测符号的数量

    # 初始化 MultinomialHMM 模型
    model = MultinomialHMM(n_components=n_states, random_state=42, n_iter=100, tol=1e-4, init_params="")

    # 设置模型的初始参数
    model.startprob_ = np.array(pi)  # 初始状态概率
    model.transmat_ = np.array(A)  # 状态转移矩阵
    model.emissionprob_ = np.array(B)  # 发射概率矩阵

    print("HMM 模型已创建：")
    print("初始状态概率 (pi):")
    print(model.startprob_)
    print("状态转移矩阵 (A):")
    print(model.transmat_)
    print("发射概率矩阵 (B):")
    print(model.emissionprob_)

    return model


# 示例用法
if __name__ == "__main__":
    # 定义初始参数
    pi = [0.2, 0.4, 0.4]  # 初始状态概率
    A = [
        [0.6, 0.3, 0.1],  # 从状态 1 转移到各状态的概率
        [0.2, 0.5, 0.3],  # 从状态 2 转移到各状态的概率
        [0.1, 0.2, 0.7]  # 从状态 3 转移到各状态的概率
    ]  # 状态转移矩阵
    B = [
        [0.5, 0.4, 0.1],  # 状态 1 发射到各观测符号的概率
        [0.3, 0.4, 0.3],  # 状态 2 发射到各观测符号的概率
        [0.2, 0.3, 0.5]  # 状态 3 发射到各观测符号的概率
    ]  # 发射概率矩阵

    # 创建并初始化 HMM 模型
    hmm_model = create_hmm_model(pi, A, B)