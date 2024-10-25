'''
无需为所有候选片段设置状态，只需要把该候选片段塞到需要等待的waits()中即可
比如A → B → C，将其初始化为(A → B → C， 1)将其塞到waits(A)中，意思是现在在等待 A 事件的出现
当事件 A 出现后，会跳转到下一个waits(B)中，并将waits(A)中的(A → B → C， 1)删除，
在 B 中为(A → B → C， 2)，下一步为(A → B → C， 3)，当状态与len(squence)相同时，频次 +1，状态清0

'''

from collections import defaultdict

bag = []

class Episode:
    def __init__(self, sequence):
        self.sequence = sequence    # 定义所有候选序列，如A-B-C，当状态完成这三个时，频率+1
        self.freq = 0

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
        waits[first_event].append((episode, 1))
        print(f"候选片段{idx} 初始化：waits({first_event}) = {(episode, 1)}")

    return waits




if __name__ == "__main__":
    candidate_episodes = [Episode(['A', 'B', 'C']), Episode(['B', 'C'])]
    event_stream = [('A', 1), ('B', 3), ('D', 4), ('C', 6), ('E', 12), ('A', 14), ('B', 15), ('C', 17)]
    lambda_min = 0.2

    waits = init_waits(candidate_episodes)

    # 初始化后的waits列表
    print("\n初始化后的 waits 状态：")
    for event, waiting_list in waits.items():
        print(f"waits({event}) = {waiting_list}")


