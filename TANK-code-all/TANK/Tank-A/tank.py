
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import config


class TankModel(nn.Module):


    def __init__(self):
        super(TankModel, self).__init__()
        # 第一层水箱参数（示例修改）
        self.outflow_rate_1_h1  = nn.Parameter(torch.rand(1) * (0.25 - 0.05) + 0.05)  # FC: 50 - 500 mm
        self.outflow_rate_1_h2 =  nn.Parameter(torch.rand(1) * (0.1 - 0.05) + 0.05)
        self.height_1_h1 =nn.Parameter(torch.rand(1) * (60 - 40) + 40)  # 第一层第一个出流出口（h1）的出流高度：5 - 10 单位
        self.height_1_h2 = nn.Parameter(torch.rand(1) * (40 - 0) + 0) # 第一层第二个出流出口（h2）的出流高度：h2 < h1
        self.inflow_rate_2 = nn.Parameter(torch.rand(1) * (0.3 - 0.01) + 0.01)

        # 第二层水箱参数
        self.outflow_rate_2 = nn.Parameter(torch.rand(1) * (0.3-0.005) + 0.005) # 第二层流出速率：0.5 - 2.5 单位/时间
        self.height_2 = nn.Parameter(torch.rand(1) * (30 - 0) + 0)  # 第二层水箱出流高度：4 - 8 单位
        self.inflow_rate_3 = nn.Parameter(torch.rand(1) * (0.2 - 0.005) + 0.005)

        # 第三层水箱参数
        self.outflow_rate_3 = nn.Parameter(torch.rand(1) * (0.15 - 0.005) +0.005) # 第三层流出速率：0.3 - 2 单位/时间
        self.height_3 = nn.Parameter(torch.rand(1) * (30 - 0) + 0) # 第三层水箱出流高度：3 - 6 单位
        self.inflow_rate_4 = nn.Parameter(torch.rand(1) * (0.1 - 0.005) +0.005)

        # 第四层水箱参数
        self.outflow_rate_4 =nn.Parameter(torch.rand(1) * (0.1 - 0.0006) +0.0006) # 第四层流出速率：0.1 - 1.5 单位/时间

    def apply_param_constraints(self):
        with torch.no_grad():
            # 第一层约束
            self.outflow_rate_1_h1.clamp_(min=0.05, max=0.25)
            self.outflow_rate_1_h2.clamp_(min=0.05, max=0.1)
            self.inflow_rate_2.clamp_(min=0.01, max=0.3)
            # 第二层约束
            self.outflow_rate_2.clamp_(min=0.005, max=0.3)
            self.inflow_rate_3 .clamp_(min=0.005, max=0.2)
            # 第三层约束
            self.outflow_rate_3.clamp_(min=0.005, max=0.15)
            self.inflow_rate_4 .clamp_(min=0.005, max=0.1)
            # 第四层约束
            self.outflow_rate_4.clamp_(min=0.0006, max=0.1)

    def forward(self, packed_P, initial_S, Area, baseflow):
        # 解包 PackedSequence
        P, lengths = rnn_utils.pad_packed_sequence(packed_P, batch_first=True)

        # 初始化状态变量
        batch_size = P.size(0)
        water_level_1 = initial_S[0].repeat(batch_size)
        water_level_2 = initial_S[1].repeat(batch_size)
        water_level_3 = initial_S[2].repeat(batch_size)
        water_level_4 = initial_S[3].repeat(batch_size)

        # 确保所有变量在相同设备
        water_level_1 = water_level_1.to(self.outflow_rate_1_h1.device)
        water_level_2 = water_level_2.to(self.outflow_rate_1_h1.device)
        water_level_3 = water_level_3.to(self.outflow_rate_1_h1.device)
        water_level_4 = water_level_4.to(self.outflow_rate_1_h1.device)
        # 用于存储每个时间步的流出量
        outflow_values = []

        # 遍历时间步
        for t in range(P.size(1)):
            # 创建一个掩码来检查哪些样本在当前时间步 t 是有效的
            valid_mask = (lengths > t).float().to(config.device)

            # 第一层水箱计算
            inflow_1 = P[:, t, :].squeeze(-1)
            # 计算第一层两个出流出口的流出量，根据水位和出流高度进行不同的计算

            outflow_1_h1 = torch.clamp_min(self.outflow_rate_1_h1 * (water_level_1 - self.height_1_h1 + inflow_1),0) * valid_mask
            outflow_1_h2 = torch.clamp_min(self.outflow_rate_1_h2 * (water_level_1 - self.height_1_h2 + inflow_1), 0) * valid_mask
            outflow_1 = (outflow_1_h1 + outflow_1_h2) * valid_mask
            outflow = torch.clamp_min(self.inflow_rate_2 * (water_level_1 + inflow_1), 0) * valid_mask

            # 更新第一层水箱水位
            water_level_1 = water_level_1 + inflow_1 * valid_mask - outflow_1 * valid_mask - outflow * valid_mask

            # 第二层水箱计算
            inflow_2 = outflow  # 第一层的流出是第二层的流入
            # 根据出流高度计算流出量
            outflow_2 = torch.clamp_min(self.outflow_rate_2 * (water_level_2 + inflow_2 - self.height_2), 0) * valid_mask
            outflow_23 = torch.clamp_min(self.inflow_rate_3 * (water_level_2 + inflow_2), 0) * valid_mask

            # 更新第二层水箱水位
            water_level_2 = water_level_2 + inflow_2 * valid_mask - outflow_2 * valid_mask - outflow_23 * valid_mask

            # 第三层水箱计算
            inflow_3 = outflow_23  # 第二层的流出是第三层的流入
            # 根据出流高度计算流出量
            outflow_3 = torch.clamp_min(self.outflow_rate_3 * (water_level_3 + inflow_3 - self.height_3),0) * valid_mask
            outflow_34 = torch.clamp_min(self.inflow_rate_4 * (water_level_3 + inflow_3), 0) * valid_mask

            # 更新第三层水箱水位
            water_level_3 = water_level_3 + inflow_3 * valid_mask - outflow_3 * valid_mask - outflow_34 * valid_mask

            # 第四层水箱计算
            inflow_4 = outflow_34  # 第三层的流出是第四层的流入
            # 根据出流高度计算流出量
            outflow_4 = torch.clamp_min(self.outflow_rate_4 * (water_level_4 + inflow_4), 0) * valid_mask

            # 更新第四层水箱水位
            water_level_4 = water_level_4 + inflow_4 * valid_mask - outflow_4 * valid_mask

            # 总流出量，可根据实际需求调整计算方式，这里简单相加
            total_outflow = (outflow_1 + outflow_2 + outflow_3 + outflow_4) * valid_mask

            # 存储流出量，考虑面积因素
            outflow_values.append(total_outflow * Area / 3.6)

        # 将结果堆叠成一个张量
        outflow_tensor = torch.stack(outflow_values, dim=1)

        return outflow_tensor