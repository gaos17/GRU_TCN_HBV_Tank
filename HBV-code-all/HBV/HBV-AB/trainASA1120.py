'''
train程序
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
from HBVASAandCMAES import HBVModel

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--zhanming", type=str, required=True)
parser.add_argument("--time", type=int, required=True)
args = parser.parse_args()

# 动态更新配置
config.zhanming = args.zhanming
config.time = args.time

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
        self.output_data = []

        # 参数范围
        self.param_ranges = {
            'FC': (50, 100),
            'L': (0.7, 1.0),
            'LP': (0.3, 0.7),
            'beta': (3.0, 6.0),
            'K0': (0.05, 0.5),
            'K1': (0.01, 0.3),
            'Kd': (0.01, 0.2),
            'K2': (0.001, 0.1),
            'gamma_a': (0.5,3),
            'gamma_tau': (0.5,8)
        }

    def _hash_params(self, params):
        """生成参数字典的稳定哈希值（处理浮点精度问题）"""
        rounded_params = {k: round(v, 6) for k, v in params.items()}  # 6位小数精度
        return hash(frozenset(sorted(rounded_params.items())))

    def evaluate(self, params, dataloader):
        """评估当前参数下的模型损失（MSE）"""
        # 复制参数以避免修改原始字典
        hourly_params = params.copy()
        # 对 K0/K1/K2/Kd 应用非线性转换
        for k in ['K0', 'K1', 'K2', 'Kd']:
            daily_value = params[k]
            hourly_params[k] = 1 - (1 - daily_value) ** (1 / 24)

        self.model.set_parameters(params)
        total_loss = 0.0
        criterion = torch.nn.MSELoss()
        for _, input, _, flowbase, area, target in dataloader:
            input = input.to(config.device)
            flowbase = flowbase.to(config.device)
            area = area.to(config.device)
            target = target.to(config.device)
            output = self.model(config.ETP, input, config.initial_SM,
                                config.initial_SUZ, config.initial_SLZ, area, flowbase)
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

    def update_temperature(self, acceptance_rate,current_loss):
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
        improvements = sum(1 for i in range(1, len(window))
                           if window[i] < window[i - 1] * 0.999)  # 0.1%以上提升
        return improvements / len(window)

    def _should_reheat(self, improvement_rate):
        """判断是否需要回温"""
        cond1 = self.current_temp < self.initial_temp * 0.2
        cond2 = improvement_rate < 0.05
        cond3 = np.random.random() < 0.05  # 10%概率
        return cond1 and cond2 and cond3

    '''
    def save_results_to_excel(self):
        
        if not self.output_data:
            return
        # 固定文件名
        filename = config.get_result_path()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 保存当前所有数据（覆盖模式）
        try:
            pd.DataFrame(self.output_data).to_excel(filename, index=False)
            # print(f"\n实时保存到: {filename}")  # 调试时可取消注释
        except Exception as e:
            print(f"保存失败: {str(e)}")
'''
    def run(self, train_loader, val_loader):
        start_time = time.time()
        # 初始化当前解
        current_params = {key: np.random.uniform(*self.param_ranges[key]) for key in self.param_ranges}

        # 记忆化检查1
        params_hash = self._hash_params(current_params)
        if params_hash in self.memory:
            current_loss = self.memory[params_hash]
        else:
            current_loss = self.evaluate(current_params, train_loader)
            self.memory[params_hash] = current_loss

        # 初始化最佳状态为当前状态
        self.best_state = current_params.copy()
        self.best_loss = current_loss

        no_improvement = 0
        for iteration in range(self.max_iter):
            should_output = (iteration < 10) or ((iteration + 1) % 1 == 0)
            if should_output:
                elapsed = time.time() - start_time
                print(f"\n=== 迭代 {iteration + 1} 次 ===")
                print(f"用时: {elapsed:.2f}秒 | 温度: {self.current_temp:.2f}")
                print("当前参数:")
                model_params = self.model.get_parameters()
                for param, value in model_params.items():
                    print(f"{param}: {value:.6f}")
                print("=" * 40)

                # 准备记录数据
                record = {
                    'iteration': iteration + 1,
                    'time': elapsed,
                    'temperature': self.current_temp,
                    'loss': current_loss,
                    'best_loss': self.best_loss,
                    # 当前参数的日尺度
                    **{f'current_day_{k}': v for k, v in model_params.items()},
                    # 当前参数的小时尺度（仅转换K0/K1/K2/Kd）
                    **{f'current_hour_{k}': 1 - (1 - v) ** (1 / 24)
                       for k, v in model_params.items()
                       if k in ['K0', 'K1', 'K2', 'Kd']},
                }

                # 添加最佳参数（确保best_state不为None）
                if self.best_state is not None:
                    record.update({
                        # 历史最优参数的日尺度
                        **{f'best_day_{k}': v for k, v in self.best_state.items()},
                        # 历史最优参数的小时尺度
                        **{f'best_hour_{k}': 1 - (1 - v) ** (1 / 24)
                           for k, v in self.best_state.items()
                           if k in ['K0', 'K1', 'K2', 'Kd']}
                    })
                else:
                    # 如果best_state为None，用当前参数填充
                    record.update({
                        **{f'best_day_{k}': v for k, v in model_params.items()},
                        **{f'best_hour_{k}': 1 - (1 - v) ** (1 / 24)
                           for k, v in model_params.items()
                           if k in ['K0', 'K1', 'K2', 'Kd']}
                    })

                self.output_data.append(record)
                #self.save_results_to_excel()

            # 生成邻域解并评估
            neighbor_params = self.generate_neighbor(current_params)
            neighbor_loss = self.evaluate(neighbor_params, train_loader)

            # 计算损失差和接受概率
            delta_loss = neighbor_loss - current_loss
            accept_prob = math.exp(-delta_loss / self.current_temp)
            # Metropolis准则决定是否接受新解
            if delta_loss < 0 or np.random.random() < accept_prob:
                current_params = neighbor_params
                # 记忆化检查2
                params_hash = self._hash_params(current_params)
                if params_hash in self.memory:
                    current_loss = self.memory[params_hash]
                else:
                    current_loss = neighbor_loss  # 直接用已计算的neighbor_loss
                    self.memory[params_hash] = current_loss
                # current_loss = neighbor_loss
                acceptance_status = "Accepted"
            else:
                acceptance_status = "Rejected"

            def save_model_fixed(model, path):
                state_dict = model.state_dict()
                fixed_state_dict = {}

                # 确保所有参数都是1维张量
                for key, value in state_dict.items():
                    if value.dim() == 0:  # 如果是标量
                        fixed_state_dict[key] = value.unsqueeze(0)
                    else:
                        fixed_state_dict[key] = value

                torch.save(fixed_state_dict, path)
                print(f"模型已保存，参数形状: {[f'{k}:{v.shape}' for k, v in fixed_state_dict.items()]}")
            # 更新最佳解
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_state = current_params.copy()
                no_improvement = 0
                # 保存模型
                self.model.set_parameters(self.best_state)
                save_model_fixed(self.model, config.get_model_path())
            else:
                no_improvement += 1
                print(f'Count{no_improvement}/{self.patience}')

            # 计算当前接受率（滑动窗口）
            if iteration == 0:
                acceptance_rate = float(acceptance_status == "Accepted")
            else:
                acceptance_rate = 0.9 * acceptance_rate + 0.1 * (acceptance_status == "Accepted")

            # 更新温度（传入当前接受率和损失）
            self.update_temperature(acceptance_rate, current_loss)

            # 打印日志
            print(f"Iter {iteration}: Temp={self.current_temp:.2f}, "
                  f"Loss={current_loss:.4f}, Best={self.best_loss:.4f}, "
                  f"Accept={acceptance_status}, Rate={acceptance_rate:.2f}")

            # 早停检查
            if (self.current_temp < self.min_temp and
                    self.reheat_count >= 3 and  # 至少尝试回温3次
                    no_improvement >= self.patience):
                break
            '''if no_improvement >= self.patience:
                print(f"Early stopping at iteration {iteration}")
                break'''

            # 终止条件
            if self.current_temp < self.min_temp:
                print(f"Reached minimum temperature at iteration {iteration}")
                break

        # 验证集评估
        val_loss = self.evaluate(self.best_state, val_loader)
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"最终温度: {self.current_temp:.4f}")
        print(f"最佳损失: {self.best_loss:.4f}")
        return self.model, self.best_state, self.best_loss


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
    model = HBVModel().to(config.device)
    model, best_params, best_loss = train(model, config.patience)
    print(f"Best Parameters: {best_params}")
    print(f"Best Loss: {best_loss:.4f}")



