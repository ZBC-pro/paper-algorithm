'''
无需为所有候选片段设置状态，只需要把该候选片段塞到需要等待的waits()中即可
比如A → B → C，将其初始化为(A → B → C， 1)将其塞到waits(A)中，意思是现在在等待 A 事件的出现
当事件 A 出现后，会跳转到下一个waits(B)中，并将waits(A)中的(A → B → C， 1)删除，
在 B 中为(A → B → C， 2)，下一步为(A → B → C， 3)，当状态与len(squence)相同时，频次 +1，状态清0

'''
# 创建多个列表的库
from collections import defaultdict
# 复杂度分析
import big_o

bag = []

class Episode:
    def __init__(self, sequence):
        self.sequence = sequence    # 定义所有候选序列，如A-B-C，当状态完成这三个时，频率+1
        self.freq = 0

    # 交互方法，将地址转化为字符串
    def __repr__(self):
        return f"Episode(sequence={self.sequence}, freq={self.freq})"

# 将所有事件类型转化为wait(事件)
# def wait_trans(event_stream):
#     waits = defaultdict(list)
#     for event, timestamp in event_stream:
#         waits[event] = []

# 初始化候选片段的等待状态
# 思路：遍历所有的候选片段，将其第一个事件和状态加入到相应的waits列表中
def init_waits(candidate_episodes):
    waits = defaultdict(list)
    for idx, episode in enumerate(candidate_episodes, start=1):
        first_event = episode.sequence[0]
        waits[first_event].append((episode, 0))
        print(f"候选片段{idx} 初始化：waits({first_event}) = {(episode, 1)}")

    return waits

# 移动事件到下一个并改变状态
def remove_events(event_stream, waits):
    for event, timestamp in event_stream:
        if event in waits:
            current_wait_list = waits[event][:]
            for episode, state in current_wait_list:
                print("\n**&&")
                print(episode, state)
                next_state = state + 1
                if next_state >= len(episode.sequence):
                    episode.freq += 1
                    print(f"{episode.sequence} 匹配完成，增加频率：{episode.freq}")

                    # 重置状态为 0，并将其加入到等待第一个事件的列表中
                    first_event = episode.sequence[0]
                    waits[first_event].append((episode, 0))
                else:
                    next_event = episode.sequence[next_state]
                    waits[next_event].append((episode, next_state))
                waits[event].remove((episode, state))
    print("\n最终的候选片段频率：")
    for episode in candidate_episodes:
        print(f"{episode.sequence}: 出现次数 {episode.freq}")

# 用于big-O的时间复杂度分析
def run_test(n):
    candidate_episodes = [Episode(['A', 'B', 'C']), Episode(['B', 'C'])]
    # 动态生成事件流，基于大小n
    event_stream = [('A', i) if i % 2 == 0 else ('B', i) for i in range(n)]
    waits = init_waits(candidate_episodes)
    remove_events(event_stream, waits)


if __name__ == "__main__":
    candidate_episodes = [Episode(['A', 'B', 'C']), Episode(['B', 'C']), Episode(['A', 'C'])]
    event_stream = [('A', 1), ('B', 3), ('D', 4), ('C', 6), ('E', 12), ('A', 14), ('B', 15), ('C', 17), ('B', 18), ('C', 20)]
    lambda_min = 0.2

    waits = init_waits(candidate_episodes)

    # 初始化后的waits列表
    print("\n初始化后的 waits 状态：")
    for event, waiting_list in waits.items():
        print(f"waits({event}) = {waiting_list}")

    # print("\n***********")
    # 遍历 waits 字典中的所有事件和等待列表
    # for event, waiting_list in waits.items():
    #     print(f"当前事件: {event}")
    #
    #     for episode, state in waiting_list:
    #         # 输出 Episode 对象和状态
    #         print(f"  Episode: {episode}, State: {state}")
    #
    #         # 如果需要单独提取 episode 的序列和频率
    #         sequence = episode.sequence
    #         freq = episode.freq
    #         print(f"  Sequence: {sequence}, Frequency: {freq}")

    remove_events(event_stream, waits)

    # 使用 big_o 的 big_o 函数进行复杂度测试
    # best, complexity = big_o.big_o(run_test, big_o.datagen.n_, n_repeats=100)
    # print(f"Estimated complexity: {best}")