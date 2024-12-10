import numpy as np

# 初始状态概率
pi = np.array([0.2, 0.4, 0.4])

# 状态转移矩阵
A = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# 发射矩阵（B），每行对应一个盒子的红白概率 [P(红), P(白)]
B = np.array([
    [0.5, 0.5],  # 盒子1
    [0.4, 0.6],  # 盒子2
    [0.7, 0.3]   # 盒子3
])

# 观测序列 {红, 白, 红}
observations = [0, 1, 0]  # 红 = 0, 白 = 1

# 前向算法实现
def forward_algorithm(pi, A, B, observations):
    N = A.shape[0]  # 状态数量
    T = len(observations)  # 观测序列长度

    # 初始化 alpha
    alpha = np.zeros((T, N))
    # 初始化 t=0
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, observations[0]]

    # 递推计算 alpha
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = sum(alpha[t-1, i] * A[i, j] for i in range(N)) * B[j, observations[t]]

    # 观测序列的总概率
    P = sum(alpha[T-1, i] for i in range(N))
    return P

# 维特比算法实现
def viterbi_algorithm(pi, A, B, observations):
    N = A.shape[0]  # 状态数量
    T = len(observations)  # 观测序列长度

    # 初始化 delta 和 psi
    delta = np.zeros((T, N))  # 最大概率
    psi = np.zeros((T, N), dtype=int)  # 记录路径

    # 初始化 t=0
    for i in range(N):
        delta[0, i] = pi[i] * B[i, observations[0]]
        psi[0, i] = 0  # 初始时没有前驱

    # 递推计算 delta 和 psi
    for t in range(1, T):
        for j in range(N):
            max_prob = -1
            max_state = -1
            for i in range(N):
                prob = delta[t-1, i] * A[i, j]
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            delta[t, j] = max_prob * B[j, observations[t]]
            psi[t, j] = max_state

    # 回溯，找出最可能的隐藏状态序列
    hidden_states = np.zeros(T, dtype=int)
    hidden_states[T-1] = np.argmax(delta[T-1, :])  # 最后一步的最大概率对应的状态
    for t in range(T-2, -1, -1):
        hidden_states[t] = psi[t+1, hidden_states[t+1]]

    return hidden_states, np.max(delta[T-1, :])

# 计算观测序列的概率（前向算法）
P_observation = forward_algorithm(pi, A, B, observations)
print(f"观测序列 {['红' if obs == 0 else '白' for obs in observations]} 的概率为: {P_observation}")

# 计算最可能的隐藏状态序列（维特比算法）
hidden_states, max_prob = viterbi_algorithm(pi, A, B, observations)
print(f"最可能的隐藏状态序列为: {hidden_states + 1}")  # 状态从1开始编号
print(f"最可能序列的概率为: {max_prob}")