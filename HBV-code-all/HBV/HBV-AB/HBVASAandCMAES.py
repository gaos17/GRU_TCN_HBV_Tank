"""
为匹配ASA算法和CMAES算法，增加了对于张量的处理
作者： gaos
日期：2024/11/15
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import config

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


class HBVModel(nn.Module):
    def __init__(self):
        super(HBVModel, self).__init__()
        self.FC = nn.Parameter(torch.rand(1) * (100 - 50) + 50)  # FC: 50 - 500 mm
        self.L = nn.Parameter(torch.rand(1) * (1.0 - 0.7) + 0.7)  # L: 0.3 - 1.0
        self.LP = nn.Parameter(torch.rand(1) * (0.7 - 0.3) + 0.3)  # LP: 0.2 - 0.7
        self.beta = nn.Parameter(torch.rand(1) * (6.0 - 3.0) + 3.0)  # beta: 1.0 - 6.0
        self.K0 = nn.Parameter(torch.rand(1) * (0.5 - 0.05) + 0.05)  # K0: 0.05 - 0.5
        self.K1 = nn.Parameter(torch.rand(1) * (0.3 - 0.01) + 0.01)  # K1: 0.01 - 0.3
        self.Kd = nn.Parameter(torch.rand(1) * (0.2 - 0.01) + 0.01)  # Kd: 0.01 - 0.2
        self.K2 = nn.Parameter(torch.rand(1) * (0.1 - 0.001) + 0.001)  # K2: 0.001 - 0.1

        # 伽马分布单位线参数
        self.gamma_a = nn.Parameter(torch.rand(1) * (3.0 - 0.5) + 0.5)  # 形状参数 a: 1.0 - 2.0
        self.gamma_tau = nn.Parameter(torch.rand(1) * (8.0 - 0.5) + 0.5)  # 时间尺度参数 tau: 1.0 - 3.0

        # 定义转换函数
        def convert_daily_to_hourly(daily_coefficient):
            # 转换为小时尺度
            return 1 - (1 - daily_coefficient) ** (1 / 24)

        # 转换为小时尺度
        self.K0 = nn.Parameter(convert_daily_to_hourly(self.K0))
        self.K1 = nn.Parameter(convert_daily_to_hourly(self.K1))
        self.Kd = nn.Parameter(convert_daily_to_hourly(self.Kd))
        self.K2 = nn.Parameter(convert_daily_to_hourly(self.K2))
    def gamma_unit_hydrograph(self, t, a, tau):  # a 控制曲线形状（a<1 时陡峭）,tau 控制时间延迟（值越大延迟越长）。
        # 伽马分布公式
        numerator = (t ** (a - 1)) * torch.exp(-t / tau)  # 模拟降雨到径流的滞后效应（汇流过程）
        denominator = torch.exp(torch.lgamma(a)) * (tau ** a)  # 使用 torch.lgamma 计算伽马函数的对数值
        return numerator / denominator
    def set_parameters(self, params):
        """
        接收遗传算法传递的参数，并更新模型的参数值。
        :param params: dict，包含模型参数的键值对。
        """
        self.FC.data = torch.tensor(params['FC'])
        self.L.data = torch.tensor(params['L'])
        self.LP.data = torch.tensor(params['LP'])
        self.beta.data = torch.tensor(params['beta'])
        self.K0.data = torch.tensor(params['K0'])
        self.K1.data = torch.tensor(params['K1'])
        self.Kd.data = torch.tensor(params['Kd'])
        self.K2.data = torch.tensor(params['K2'])
        self.gamma_a.data = torch.tensor(params['gamma_a'])
        self.gamma_tau.data = torch.tensor(params['gamma_tau'])

    def get_parameters(self):
        """
        返回当前模型参数的字典，用于遗传算法提取种群参数。
        """
        return {
            'FC': self.FC.item(),
            'L': self.L.item(),
            'LP': self.LP.item(),
            'beta': self.beta.item(),
            'K0': self.K0.item(),
            'K1': self.K1.item(),
            'Kd': self.Kd.item(),
            'K2': self.K2.item(),
            'gamma_a': self.gamma_a.item(),
            'gamma_tau': self.gamma_tau.item()
        }



    def forward(self, ETP, packed_P, initial_SM, initial_SUZ, initial_SLZ, Area, baseflow):
        # 解包PackedSequence
        P, lengths = rnn_utils.pad_packed_sequence(packed_P, batch_first=True)

        # 初始化状态变量
        batch_size = P.size(0)
        SM = initial_SM.repeat(batch_size)
        SUZ = initial_SUZ.repeat(batch_size)
        SLZ = initial_SLZ.repeat(batch_size)
        ETP = ETP.repeat(P.size(1))

        Q_values = []

        # 遍历时间步
        for t in range(P.size(1)):
            # 创建一个掩码来检查哪些样本在当前时间步 t 是有效的
            valid_mask = (lengths > t).float().to(config.device)
            # 1. 使用当前时刻的降雨量 Pt 更新状态变量
            Pt = P[:, t, :].squeeze(-1)  # 当前时刻的降雨量
            # 蒸散发计算
            ET = ETP[t] * torch.min(SM / (self.FC * self.LP), torch.tensor(1.0, device=config.device))
            # 快流产流系数
            R = (SM / self.FC) ** self.beta
            # 更新土壤湿度
            SM = torch.clamp(SM + (Pt - ET - R) * valid_mask, min=0)  # 更新土壤湿度
            # 更新上层储水量
            Qd = self.Kd * SUZ  # 中层流出
            SUZ = torch.clamp(SUZ + (R - self.K0 * torch.max(SUZ - self.L, torch.tensor(0.0, device=config.device))
                                     - self.K1 * SUZ - Qd) * valid_mask, min=0)  # 更新上层存储
            # 更新下层储水量
            SLZ = torch.clamp(SLZ + (Qd - self.K2 * SLZ) * valid_mask, min=0)  # 更新下层存储
            # 2. 基于更新后的状态变量计算流量 Q_t
            Q0 = self.K0 * torch.max(SUZ - self.L, torch.tensor(0.0, device=config.device))  # 超渗产流
            Q1 = self.K1 * SUZ  # 上层流出
            Q2 = self.K2 * SLZ  # 渗漏或基流

            # 计算当前时刻流量 Q_t（逻辑归属下一时刻 t+1）
            Q = (Q0 + Q1 + Q2) * valid_mask
            Q_values.append(Q)

        # 将结果堆叠成一个张量
        Q_tensor = torch.stack(Q_values, dim=1)  # Shape: (batch_size, timesteps)

        # 对径流进行卷积计算
        t_max = int(3 * self.gamma_a * self.gamma_tau)
        # 计算伽马分布单位线权重
        time_steps_full = torch.arange(1, t_max + 2, device=config.device, dtype=torch.float32)  # 时间步
        gamma_weights_full = self.gamma_unit_hydrograph(time_steps_full, self.gamma_a,
                                                        self.gamma_tau)  # Shape: (seq_length,)
        # 对伽马分布权重进行归一化
        gamma_weights = gamma_weights_full / gamma_weights_full.sum()
        # 初始化输出张量，用于存储卷积结果
        Q_weighted = []
        for i in range(batch_size):
            # 获取当前样本的实际序列长度
            seq_length = lengths[i].item()  # 当前样本的有效时间步长度
            # 获取当前样本的降雨径流序列
            Q_sample = Q_tensor[i, :seq_length]  # Shape: (seq_length,)
            # 初始化输出流量序列
            Q_convolved = torch.zeros(seq_length)  # Shape: (seq_length,)
            # 逐步累积计算卷积
            for t in range(seq_length):
                # 确定当前时间步需要的权重和流量范围
                start_idx = max(0, t - t_max)  # 流量的起始索引
                end_idx = t + 1  # 流量的结束索引（不包括）
                Q_segment = Q_sample[start_idx:end_idx]  # 取最近的流量值
                weights_segment = gamma_weights[:len(Q_segment)]  # 对应的权重值
                # 反转流量值和权重顺序进行相乘
                Q_convolved[t] = torch.sum(Q_segment.flip(0) * weights_segment)
            # 将结果填充到与最大序列长度一致
            Q_padded = F.pad(Q_convolved, (0, Q_tensor.size(1) - seq_length))  # Shape: (timesteps,)
            Q_weighted.append(Q_padded)

        # 将结果堆叠成一个张量
        Q_weighted_tensor = torch.stack(Q_weighted, dim=0)  # Shape: (batch_size, timesteps)

        # 加上面积和基流的考虑
        Area = Area.unsqueeze(1)  # Shape: (batch_size, 1)
        baseflow = baseflow.unsqueeze(1)  # Shape: (batch_size, 1)
        # 构造掩码，仅对有效序列长度进行计算
        mask = torch.arange(Q_weighted_tensor.size(1), device=config.device).unsqueeze(0) < lengths.unsqueeze(
            1)  # Shape: (batch_size, timesteps)
        mask = mask.float()  # 转换为浮点型，方便后续计算
        # 仅对有效部分加上基流，填充部分保持为 0
        Q_final = (Q_weighted_tensor * Area / 3.6 + baseflow) * mask

        return Q_final
