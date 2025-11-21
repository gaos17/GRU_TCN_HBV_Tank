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

class NumDataset(Dataset):
    def __init__(self, dataset1 = 'train'):
        raindistrain = pd.read_csv(f'..\\SISO-event\\{config.zhanming}\\raindistrain' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                   index_col=0)
        raindistest = pd.read_csv(f'..\\SISO-event\\{config.zhanming}\\raindistest' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                  index_col=0)
        raindisval = pd.read_csv(f'..\\SISO-event\\{config.zhanming}\\raindisval' + str(config.timesteps) + '_' + str(config.leadtimes) + '.csv',
                                 index_col=0)
        if dataset1 == 'train':
            rd = raindistrain.values
        if dataset1 == 'test':
            rd = raindistest.values
        if dataset1 == 'val':
            rd = raindisval.values
        self.data = rd.reshape((rd.shape[0] // config.timesteps, config.timesteps, rd.shape[1]))

    def __getitem__(self, item):
        #不同模型的lable不完全一样,需要根据模型做具体的修改
        #time只取了timesteps最后一个维度的
        time = self.data[item,-1,:2 ]
        input = self.data[item,:,2:3 ]
        flowbase = self.data[item,-1, 2+config.site ]
        area = self.data[item,-1, 3+config.site ]
        lable = self.data[item,-1, 5+config.site]
        return time, input, flowbase, area,lable,

    def __len__(self):
        return self.data.shape[0]
def collate_fn(batch):
    time, input, flowbase, area, lable = zip(*batch)
    input = np.array(input).astype(float)
    input = torch.FloatTensor(input)
    flowbase = torch.FloatTensor(flowbase)
    area = torch.FloatTensor(area)
    lable = torch.FloatTensor(lable)
    return time, input, flowbase,area, lable

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
