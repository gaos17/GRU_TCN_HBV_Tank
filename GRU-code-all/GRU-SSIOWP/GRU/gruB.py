"""

作者： gaoshuai
日期： 2021年12月17日
"""

import torch.nn as nn
import torch
import config
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Attenmodel(nn.Module):
    def __init__(self):
        super(Attenmodel, self).__init__()
        self.gru = nn.GRU(input_size=config.site,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.fa = nn.Linear(config.hidden_size, 1)

    def forward(self, input, flowbase, area):
        # 解包PackedSequence
        output, lengths = pad_packed_sequence(self.gru(input.float())[0], batch_first=True)

        # 通过全连接层逐时间步处理输出
        sequence_output = self.fa(output).squeeze(-1)

        # 获取批次大小和最大序列长度
        batch_size, max_len = sequence_output.size(0), sequence_output.size(1)

        # 扩展 flowbase 和 area 的维度以匹配序列长度
        flowbase = flowbase.unsqueeze(1).expand(batch_size, max_len)
        if area.dim() == 1:
            area = area.unsqueeze(1).expand(batch_size, max_len)

        # 生成掩码矩阵，根据每个序列的有效长度
        valid_mask = torch.arange(max_len, device=sequence_output.device).unsqueeze(0) < lengths.unsqueeze(1)
        valid_mask = valid_mask.float()  # 转换为浮点数以便参与计算

        # 对每个时间步加上对应的 flowbase，并使用掩码
        output_with_flowbase = (sequence_output + flowbase) * valid_mask

        return output_with_flowbase



if __name__ =="__main__":
    net = Attenmodel()
    print(net)
    param = net.named_parameters()
    for i in param:
        print(i)