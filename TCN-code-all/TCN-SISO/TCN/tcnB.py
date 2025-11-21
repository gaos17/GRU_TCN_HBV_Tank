"""
作者： gaoshuai
日期： 2025年4月23日
修改：添加TCN实现，使用前向padding
"""

import torch.nn as nn
import torch
import config
import math


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()
        # 计算需要的padding大小（只在序列前端padding）
        self.padding = (kernel_size - 1) * dilation

        # 使用自定义的padding模式：在序列前端补零
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.relu1 = nn.ReLU()

        self.net = nn.Sequential(
            self.relu1
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 在序列前端添加padding
        padding = torch.zeros(x.shape[0], x.shape[1], self.padding, device=x.device)
        x_padded = torch.cat((padding, x), dim=2)

        # 通过卷积层
        out = self.conv1(x_padded)
        out = self.net(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Attenmodel(nn.Module):
    def __init__(self):
        super(Attenmodel, self).__init__()
        # TCN参数
        num_channels = [16, 16, 16, 16, 16]  # TCN中每层的通道数,这个参数的个数对结果影响极大,太少不行.
        kernel_size = 3

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size)]

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, input, flowbase, area):
        # input shape: [batch, timesteps, channels]
        input = input.permute(0, 2, 1)  # -> [batch, channels, timesteps]

        # TCN layers
        output = self.tcn(input)

        # 只取最后一个时间步的结果
        output = output[:, :, -1]  # [batch_size, channels]
        output = self.fc(output)

        # 应用面积和基流量影响
        output = output.squeeze(1) + flowbase

        return output


if __name__ == "__main__":
    net = Attenmodel()
    print(net)
    param = net.named_parameters()
    # for i in param:
    #     print(i)