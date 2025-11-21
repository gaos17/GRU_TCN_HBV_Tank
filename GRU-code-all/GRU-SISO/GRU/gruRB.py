"""

作者： gaoshuai
日期： 2021年12月17日
"""

import torch.nn as nn
import torch
import config

class Attenmodel(nn.Module):
    def __init__(self):
        super(Attenmodel, self).__init__()
        self.gru = nn.GRU(input_size=config.site,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.fa = nn.Linear(config.hidden_size, 1)
        self.A = nn.Parameter(torch.rand(1) * 1000)



    def forward(self, input, flowbase, area):
        _, hidden = self.gru(input) #忽略输出序列，仅保留最后一个隐藏状态

        output = self.fa(hidden.squeeze(0)).squeeze(1) *self.A + flowbase #去除隐藏状态中的第一个维度和输出的第二个维度
        return output

if __name__ =="__main__":
    net = Attenmodel()
    print(net)
    param = net.named_parameters()
    for i in param:
        print(i)