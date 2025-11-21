'''
train程序 (修改版 - 直接使用TankModel0721)
'''
import argparse
import os
from dataset import get_dataloader
import torch
import config
import numpy as np
import math
import time
import pandas as pd
import plotresults as pr
from tankmodel0721 import TankModel  # 直接导入您的TankModel
import torch.nn.utils.rnn as rnn_utils
import xlsxwriter  # 用于设置列宽
from datetime import datetime  # 用于生成日期字符串

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--zhanming", type=str, required=True)
parser.add_argument("--time", type=int, required=True)
args = parser.parse_args()

# 动态更新配置
config.zhanming = args.zhanming
config.time = args.time

# 处理预测结果格式
def process_predictions(outputs, target, times):
    """
    处理预测结果，格式化为要求的五列格式
    返回DataFrame: 索引列 | times | rainmean | real_1 | pred_1
    """
    # 初始化空列表存储结果
    all_results = []

    # 修复错误：使用detach()分离梯度
    outputs = outputs.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    # 处理每个时间序列
    for i in range(len(times)):
        # 提取时间序列信息
        time_array = times[i]
        # 转换为datetime并跳过前30个时间步（预热期）
        datetimes = pd.to_datetime(time_array[:, 0][30:], errors='coerce')
        # 提取降雨数据并跳过前30个时间步
        rain_data = time_array[:, 1].astype(float)[30:]

        # 提取真实值和预测值（已跳过前30步）
        target_values = target[i, :].flatten()
        pred_values = outputs[i, :].flatten()

        # 检查维度一致性
        n_steps = min(len(datetimes), len(rain_data), len(target_values), len(pred_values))
        datetimes = datetimes[:n_steps]
        rain_data = rain_data[:n_steps]
        target_values = target_values[:n_steps]
        pred_values = pred_values[:n_steps]

        # 创建当前事件的结果DataFrame
        event_df = pd.DataFrame({
            'times': datetimes,
            'rainmean': rain_data,
            'real_1': target_values,
            'pred_1': pred_values
        })

        # 添加到总结果
        all_results.append(event_df)

    # 合并所有事件的结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # 清除目标值和预测值均为0的多余行
        final_df = final_df[~((final_df['real_1'] == 0) & (final_df['pred_1'] == 0))]

        # 事件分割（如eval.py中所示）
        time_diff = final_df['times'].diff().dt.total_seconds()
        event_start = (time_diff > 3600) | (time_diff.isna())
        event_ids = event_start.cumsum()

        # 为每个事件重新编号索引
        final_df['new_index'] = final_df.groupby(event_ids).cumcount()
        final_df.set_index('new_index', inplace=True)
        final_df.index.name = None
        return final_df
    return pd.DataFrame()

class AdaptiveSimulatedAnnealing:
    def __init__(self, model):
        self.model = model
        self.best_state = None
        self.best_loss = float('inf')
        self.current_temp = self.initial_temp = 180.0  # 初始温度
        self.min_temp = 1e-3             # 终止温度
        self.cooling_rate = 0.96         # 基础冷却速率
        self.adaptive_factor = 1.05      # 自适应调整因子
        self.max_iter = 500              # 最大迭代次数
        self.patience = config.patience  # 早停耐心值
        self.l1_lambda = 0.01   # L1正则化强度
        self.l2_lambda = 0.01   # L2正则化强度
        # 记忆化存储系统
        self.memory = {}      # 格式: {参数哈希值: 损失值}
        self.memory_hits = 0  # 统计命中次数
        # 动态温度控制参数
        self.last_improvement = 0     # 上次改进的迭代次数
        self.improvement_window = 30  # 改进率计算窗口
        self.reheat_base_temp = None  # 回温基准温度
        self.reheat_count = 0         # 回温次数统计
        # 用于跟踪数据集类型
        self.current_dataset = 'train'  # 默认训练集

        # 参数范围 - 根据您的TankModel调整
        self.param_ranges = {
            'outflow_rate_1_h1': (0.05, 0.25),
            'outflow_rate_1_h2': (0.05, 0.1),
            'height_1_h1': (40.0, 60.0),
            'height_1_h2': (0.0, 40.0),
            'inflow_rate_2': (0.01, 0.3),
            'outflow_rate_2': (0.005, 0.3),
            'inflow_rate_3': (0.005, 0.2),
            'outflow_rate_3': (0.005, 0.15),
            'inflow_rate_4': (0.005, 0.1),
            'outflow_rate_4': (0.0006, 0.1),
        }

    def _hash_params(self, params):
        """生成参数字典的稳定哈希值（处理浮点精度问题）"""
        rounded_params = {k: round(v, 6) for k, v in params.items()}  # 6位小数精度
        return hash(frozenset(sorted(rounded_params.items())))

    def evaluate(self, params, dataloader, dataset_type=None):
        """评估当前参数下的模型损失（MSE）"""
        # 复制参数以避免修改原始字典
        hourly_params = params.copy()
        # 对 K0/K1/K2/Kd 应用非线性转换（如果存在）
        for k in ['K0', 'K1', 'K2', 'Kd']:
            if k in params:
                daily_value = params[k]
                hourly_params[k] = 1 - (1 - daily_value) ** (1 / 24)

        # 使用 set_parameters 方法设置模型参数
        self.model.set_parameters(params)

        # 设置当前数据集类型
        if dataset_type:
            self.current_dataset = dataset_type

        total_loss = 0.0
        criterion = torch.nn.MSELoss()
        for times, input, _, flowbase, area, target in dataloader:
            # ========== 关键修改 ========== #
            # 确保batch_sizes正确处理
            batch_size = input.batch_sizes[0]  # 直接取值，无需.item()

            # 将数据移动到设备
            flowbase = flowbase.to(config.device)
            area = area.to(config.device)
            target = target.to(config.device)

            # 转换batch_sizes为张量
            if isinstance(input.batch_sizes, tuple):
                batch_sizes = torch.tensor(input.batch_sizes, dtype=torch.int64)
            else:
                batch_sizes = input.batch_sizes

            # 重构PackedSequence
            input = rnn_utils.PackedSequence(
                data=input.data.to(config.device),
                batch_sizes=batch_sizes.to(config.device),  # 确保为张量
                sorted_indices=input.sorted_indices,
                unsorted_indices=input.unsorted_indices
            )
            # ========== 修改结束 ========== #

            # 调用模型 - 匹配您的TankModel参数签名
            output = self.model(
                input,
                config.initial_SM,  # 初始状态
                area,                # 区域面积
                flowbase             # 基流
            )  # 移除多余的参数

            mse_loss = criterion(output[:, 29:], target[:, 29:])

            # 计算L1/L2正则化项
            l1_penalty = 0.0
            l2_penalty = 0.0
            for param in self.model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
                l2_penalty += torch.sum(param ** 2)
            # 组合损失
            total_loss += mse_loss.item() + self.l1_lambda * l1_penalty.item() + self.l2_lambda * l2_penalty.item()

        return total_loss / len(dataloader)

    # 新增：获取预测数据方法
    def get_predictions(self, dataloader, dataset_type):
        """获取五列表格格式的预测数据"""
        self.model.set_parameters(self.best_state)
        self.current_dataset = dataset_type
        predictions = pd.DataFrame()

        with torch.no_grad():
            for times, input, _, flowbase, area, target in dataloader:
                # ========== 数据处理 ========== #
                batch_size = input.batch_sizes[0]
                flowbase = flowbase.to(config.device)
                area = area.to(config.device)
                target = target.to(config.device)

                if isinstance(input.batch_sizes, tuple):
                    batch_sizes = torch.tensor(input.batch_sizes, dtype=torch.int64)
                else:
                    batch_sizes = input.batch_sizes

                input = rnn_utils.PackedSequence(
                    data=input.data.to(config.device),
                    batch_sizes=batch_sizes.to(config.device),
                    sorted_indices=input.sorted_indices,
                    unsorted_indices=input.unsorted_indices
                )
                # ==============================

                # 模型预测
                output = self.model(
                    input,
                    config.initial_SM,
                    area,
                    flowbase
                )

                # 处理预测结果
                batch_predictions = process_predictions(output[:, 29:], target[:, 29:], times)
                # 添加数据集类型标记
                batch_predictions['dataset'] = dataset_type
                predictions = pd.concat([predictions, batch_predictions], ignore_index=False)

        return predictions

    def generate_neighbor(self, current_params):
        """生成邻域解（基于当前温度和参数范围）"""
        neighbor = {}
        for key in current_params:
            # 自适应步长：温度越高，扰动范围越大
            range_width = self.param_ranges[key][1] - self.param_ranges[key][0]
            step_size = self.current_temp * range_width * 0.1
            neighbor[key] = current_params[key] + np.random.uniform(-step_size, step_size)
            # 确保参数在合法范围内
            neighbor[key] = np.clip(neighbor[key], *self.param_ranges[key])
        return neighbor

    def update_temperature(self, acceptance_rate, current_loss):
        """自适应调整温度（根据接受率动态调整冷却速率）"""
        # 计算近期改进率
        if acceptance_rate > 0.5:
            # 接受率过高，加快冷却以收敛
            self.current_temp *= self.cooling_rate
        else:
            # 接受率过低，减缓冷却以探索
            self.current_temp *= (self.cooling_rate ** (1 / self.adaptive_factor))

    def _get_improvement_rate(self, current_loss):
        """计算最近20代的改进率"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []

        # 记录当前损失
        self.loss_history.append(current_loss)
        window = self.loss_history[-self.improvement_window:]

        # 计算改进次数
        improvements = 0
        for i in range(1, len(window)):
            if window[i] < window[i - 1] * 0.999:  # 0.1%以上提升
                improvements += 1

        return improvements / len(window) if window else 0.0

    def _should_reheat(self, improvement_rate):
        """判断是否需要回温"""
        cond1 = self.current_temp < self.initial_temp * 0.2
        cond2 = improvement_rate < 0.05
        cond3 = np.random.random() < 0.05  # 10%概率
        return cond1 and cond2 and cond3

    def run(self, train_loader, val_loader):
        start_time = time.time()
        # 初始化当前解
        current_params = {key: np.random.uniform(*self.param_ranges[key]) for key in self.param_ranges}

        # 获取初始损失
        params_hash = self._hash_params(current_params)
        if params_hash in self.memory:
            current_loss = self.memory[params_hash]
        else:
            current_loss = self.evaluate(current_params, train_loader, 'train')
            self.memory[params_hash] = current_loss

        # 关键修复: 初始化最优状态
        self.best_loss = current_loss
        self.best_state = current_params.copy()  # 设置初始解为第一个最优解
        no_improvement = 0

        # 保存初始模型
        self.model.set_parameters(self.best_state)
        torch.save(self.model.state_dict(), config.get_model_path())

        for iteration in range(self.max_iter):
            should_output = (iteration < 10) or ((iteration + 1) % 1 == 0)
            if should_output:
                elapsed = time.time() - start_time
                print(f"\n=== 迭代 {iteration + 1} 次 ===")
                print(f"用时: {elapsed:.2f}秒 | 温度: {self.current_temp:.2f}")
                print("当前参数:")
                # 使用 get_parameters 方法获取模型参数
                model_params = self.model.get_parameters()
                for param, value in model_params.items():
                    print(f"{param}: {value:.6f}")
                print("=" * 40)

            # 生成邻域解并评估
            neighbor_params = self.generate_neighbor(current_params)
            neighbor_loss = self.evaluate(neighbor_params, train_loader, 'train')

            # 计算损失差和接受概率
            delta_loss = neighbor_loss - current_loss

            # 安全地计算指数值
            exponent = -delta_loss / self.current_temp

            # 设置指数的上限和下限
            if exponent > 700:
                accept_prob = 1.0
            elif exponent < -700:
                accept_prob = 0.0
            else:
                try:
                    accept_prob = math.exp(exponent)
                except:
                    accept_prob = 0.0

            # Metropolis准则决定是否接受新解
            if delta_loss < 0 or np.random.random() < accept_prob:
                current_params = neighbor_params
                # 记忆化检查
                params_hash = self._hash_params(current_params)
                if params_hash in self.memory:
                    current_loss = self.memory[params_hash]
                else:
                    current_loss = neighbor_loss
                    self.memory[params_hash] = current_loss
                acceptance_status = "Accepted"
            else:
                acceptance_status = "Rejected"

            # 更新最佳解
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_state = current_params.copy()
                no_improvement = 0
                # 保存模型
                self.model.set_parameters(self.best_state)
                torch.save(self.model.state_dict(), config.get_model_path())
            else:
                no_improvement += 1
                print(f'Count{no_improvement}/{self.patience}')

            # 计算当前接受率
            if iteration == 0:
                acceptance_rate = float(acceptance_status == "Accepted")
            else:
                acceptance_rate = 0.9 * acceptance_rate + 0.1 * (acceptance_status == "Accepted")

            # 更新温度
            self.update_temperature(acceptance_rate, current_loss)

            # 打印日志
            print(f"Iter {iteration}: Temp={self.current_temp:.2f}, "
                  f"Loss={current_loss:.4f}, Best={self.best_loss:.4f}, "
                  f"Accept={acceptance_status}, Rate={acceptance_rate:.2f}")

            # 早停检查
            if (self.current_temp < self.min_temp and
                    self.reheat_count >= 3 and
                    no_improvement >= self.patience):
                break

            # 终止条件
            if self.current_temp < self.min_temp:
                print(f"Reached minimum temperature at iteration {iteration}")
                break

        # 验证集评估
        val_loss = self.evaluate(self.best_state, val_loader, 'val')
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"最终温度: {self.current_temp:.4f}")
        print(f"最佳损失: {self.best_loss:.4f}")

        # ===== 新增：训练结束后保存预测结果 =====
        # 获取训练集和验证集的预测结果
        train_predictions = self.get_predictions(train_loader, 'train')
        val_predictions = self.get_predictions(val_loader, 'val')

        # 合并结果
        combined_predictions = pd.concat([train_predictions, val_predictions], ignore_index=False)

        # 保存到config指定的路径
        self.save_combined_predictions(combined_predictions)

        return self.model, self.best_state, self.best_loss

    def save_combined_predictions(self, predictions):
        """保存合并后的预测结果到config指定的路径"""
        if predictions is None or predictions.empty:
            print("警告：无预测结果可保存")
            return

        # 获取保存路径
        datestr = datetime.now().strftime("%Y%m%d")
        save_path = f"./result/{config.zhanming}/result{datestr}{config.time}.xlsx"

        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 添加自动调整列宽功能
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            predictions.to_excel(writer, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']

            # 设置时间列宽度（第一列）
            worksheet.set_column('A:A', 20)

            # 设置其他列宽度
            for i, col in enumerate(predictions.columns[1:], start=1):
                worksheet.set_column(i, i, 12)

        print(f"已保存合并预测结果至: {save_path}")


def train(model, max_iter=300, patience=50, l1_lambda=0.01, l2_lambda=0.01):
    train_loader = get_dataloader(dataset1='train')
    val_loader = get_dataloader(dataset1='val')

    asa = AdaptiveSimulatedAnnealing(model)
    # 设置正则化系数
    asa.l1_lambda = l1_lambda
    asa.l2_lambda = l2_lambda
    model, best_params, best_loss = asa.run(train_loader, val_loader)
    return model, best_params, best_loss


if __name__ == '__main__':
    # 直接使用您的TankModel类
    model = TankModel().to(config.device)
    model, best_params, best_loss = train(model, config.patience)
    print(f"Best Parameters: {best_params}")
    print(f"Best Loss: {best_loss:.4f}")