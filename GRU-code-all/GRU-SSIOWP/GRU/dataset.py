'''
准备数据集, 导入数据, 并且只对输入数据进行归一化
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
import os
import datadeal
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class NumDataset(Dataset):
    def __init__(self, dataset1 = 'train'):
        raindistrain = pd.read_csv(f'..\\..\\..\\event\\SSIOWP\\{config.zhanming}\\raindistrain' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                   index_col=0)
        raindistest = pd.read_csv(f'..\\..\\..\\event\\SSIOWP\\{config.zhanming}\\raindistest' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                  index_col=0)
        raindisval = pd.read_csv(f'..\\..\\..\\event\\SSIOWP\\{config.zhanming}\\raindisval' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                 index_col=0)
        if dataset1 == 'train':
            rd = raindistrain
        if dataset1 == 'test':
            rd = raindistest
        if dataset1 == 'val':
            rd = raindisval

        self.sequences = []
        for filename in rd['filename'].unique():
            sequence_data = rd[rd['filename'] == filename]
            # 提取时间列
            time_data = sequence_data.iloc[:, 0:2].values  # 假设时间列是前两列
            # 提取数值数据并转换为浮点数，保留最后一列为原始格式
            numeric_data = sequence_data.iloc[:, 2:-1].values.astype(float)  # 转换中间的数值列
            event_id_data = sequence_data.iloc[:, -1].values.reshape(-1, 1)  # 保留最后一列为原始格式
            # 组合时间、数值数据和事件标识符
            combined_data = np.hstack((time_data, numeric_data, event_id_data))
            self.sequences.append(combined_data)

    def __getitem__(self, item):
        sequence = self.sequences[item]
        time = sequence[:, :2]
        input = sequence[:, 2:2 + config.site].astype(np.float32)
        flowbase = sequence[0, 2 + config.site]
        area = sequence[0, 3 + config.site]
        label = sequence[:, 5 + config.site].astype(np.float32)
        return time, input, flowbase, area, label

    def __len__(self):
        return len(self.sequences)

def collate_fn(batch):
    times, inputs, flowbases, area, labels = zip(*batch)

    inputs_tensors = [torch.tensor(inp, dtype=torch.float) for inp in inputs]
    inputs_padded = pad_sequence(inputs_tensors, batch_first=True)

    # 将标签序列转换为张量并填充
    labels_tensors = [torch.tensor(lbl, dtype=torch.float) for lbl in labels]
    labels_padded = pad_sequence(labels_tensors, batch_first=True)

    # 获取每个序列的实际长度
    lengths = torch.tensor([len(inp) for inp in inputs], dtype=torch.long)


    # 排序序列以适应 pack_padded_sequence 的要求 （从长到短）
    lengths, perm_idx = lengths.sort(0, descending=True)
    inputs_padded = inputs_padded[perm_idx]
    labels_padded = labels_padded[perm_idx]
    times = [times[i] for i in perm_idx]
    flowbase = torch.FloatTensor(flowbases).index_select(0, perm_idx)
    area = torch.FloatTensor(area).index_select(0, perm_idx)

    # 打包输入序列
    packed_inputs = pack_padded_sequence(inputs_padded, lengths, batch_first=True)

    return times, packed_inputs, lengths, flowbase, area, labels_padded

def get_dataloader(dataset1 = 'train'):
    batch_size = config.batch_size
    if dataset1 == 'train':
        shuffle_value = True
    else:
        shuffle_value = False
    return DataLoader(NumDataset(dataset1),batch_size=batch_size, shuffle=shuffle_value,collate_fn=collate_fn)


if __name__ == '__main__':
    for j, (time, input, flowbase,area, target) in enumerate(get_dataloader(dataset1='val')):
        # input: [batch_size, timesteps, feature]
        # target: [batch_size, timesteps, leadtimes]
        if j == 0:
            print(j)
            # print(input)
            print(input)
            print(flowbase)
            # print(target)
            print(target)
            break
