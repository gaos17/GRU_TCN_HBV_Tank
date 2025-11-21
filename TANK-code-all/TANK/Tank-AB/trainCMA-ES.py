'''
CMA-ES for TankModel
'''
import argparse
from dataset import get_dataloader
import torch
import config
import numpy as np
import cma  # 使用 cma 库替代 cmaes
import time
import pandas as pd
from tankmodel0721 import TankModel
import torch.nn.utils.rnn as rnn_utils
from datetime import datetime
import os as os

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--zhanming", type=str, required=True)
parser.add_argument("--time", type=int, required=True)
args = parser.parse_args()

# 动态更新配置
config.zhanming = args.zhanming
config.time = args.time

# CMA-ES 超参数 (修改为适合水箱模型的设置)
MAX_GENERATIONS = 150  # 最大迭代代数
INITIAL_SIGMA = 0.2  # 初始步长
POPULATION_SIZE = 50  # 种群大小
L1_LAMBDA = 0.005  # L1正则化系数
L2_LAMBDA = 0.01  # L2正则化系数

# 水箱模型参数范围 - 与ASA训练代码中的param_ranges匹配
PARAM_RANGES = {
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


# 将参数列表转换为字典
def params_to_dict(params):
    keys = list(PARAM_RANGES.keys())

    # 确保 params 是数组类型
    if not isinstance(params, (list, tuple, np.ndarray)):
        params = [params]  # 如果是标量，转换为单元素列表

    # 检查参数数量是否匹配
    if len(params) != len(keys):
        raise ValueError(f"参数数量不匹配: 期望 {len(keys)} 个参数，但得到 {len(params)}")

    return {keys[i]: params[i] for i in range(len(keys))}


# 将字典转换为参数列表
def dict_to_params(params_dict):
    keys = list(PARAM_RANGES.keys())
    return [params_dict[key] for key in keys]


# 动态调整正则化系数
def get_dynamic_lambda(generation, max_generations):
    # 随着迭代进行逐渐减小正则化强度
    decay_factor = 1 - (generation / max_generations) ** 0.5
    return L1_LAMBDA * decay_factor, L2_LAMBDA * decay_factor


# 处理预测结果格式（从ASA训练代码复制）
def process_predictions(outputs, target, times):
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


# 计算适应度（基于ASA的评估逻辑）
# 修改后的 evaluate_fitness 函数
def evaluate_fitness(model, params, dataloader, generation=None, max_generations=None, use_regularization=True):
    # 转换为参数字典并设置模型
    params_dict = params_to_dict(params)
    model.set_parameters(params_dict)

    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    device = config.device

    # 动态正则化系数
    if generation is not None and max_generations is not None:
        l1_lambda, l2_lambda = get_dynamic_lambda(generation, max_generations)
    else:
        l1_lambda, l2_lambda = L1_LAMBDA, L2_LAMBDA

    # 遍历数据加载器
    for times, input, _, flowbase, area, target in dataloader:
        # 数据处理（来自ASA训练代码）
        batch_size = input.batch_sizes[0]
        flowbase = flowbase.to(device)
        area = area.to(device)
        target = target.to(device)

        # 转换batch_sizes为张量
        if isinstance(input.batch_sizes, tuple):
            batch_sizes = torch.tensor(input.batch_sizes, dtype=torch.int64)
        else:
            batch_sizes = input.batch_sizes

        input = rnn_utils.PackedSequence(
            data=input.data.to(device),
            batch_sizes=batch_sizes.to(device),
            sorted_indices=input.sorted_indices,
            unsorted_indices=input.unsorted_indices
        )

        # 模型前向传播
        output = model(input, config.initial_SM, area, flowbase)

        # 计算MSE损失（跳过前30个时间步预热期）
        mse_loss = criterion(output[:, 29:], target[:, 29:])

        # 计算正则化项（仅在需要时）
        reg_loss = 0.0
        if use_regularization:
            l1_penalty = 0.0
            l2_penalty = 0.0
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
                l2_penalty += torch.sum(param ** 2)
            reg_loss = l1_lambda * l1_penalty.item() + l2_lambda * l2_penalty.item()

        total_loss += mse_loss.item() + reg_loss

    return total_loss / len(dataloader)


# 保存预测结果（基于ASA代码修改）
def save_predictions(model, params_dict, dataset_type):
    model.set_parameters(params_dict)
    predictions = pd.DataFrame()
    dataloader = get_dataloader(dataset1=dataset_type)

    with torch.no_grad():
        for times, input, _, flowbase, area, target in dataloader:
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

            output = model(input, config.initial_SM, area, flowbase)
            batch_predictions = process_predictions(output[:, 29:], target[:, 29:], times)
            batch_predictions['dataset'] = dataset_type
            predictions = pd.concat([predictions, batch_predictions], ignore_index=False)

    return predictions


# CMA-ES 算法训练
def train(model, max_generations=MAX_GENERATIONS):
    results = []
    start_time = time.time()
    train_dataloader = get_dataloader(dataset1='train')
    val_dataloader = get_dataloader(dataset1='val')

    # 初始化参数（在参数范围内随机采样）
    initial_params = np.array([np.random.uniform(low, high) for (low, high) in PARAM_RANGES.values()])

    # 参数边界
    bounds = np.array(list(PARAM_RANGES.values()))
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    # 初始化 CMA-ES 优化器 - 使用 cma 库
    es = cma.CMAEvolutionStrategy(
        initial_params,  # 初始参数向量
        INITIAL_SIGMA,   # 初始步长
        {
            'bounds': [lower_bounds, upper_bounds],  # 下界和上界
            'popsize': POPULATION_SIZE,
        }
    )

    best_params = None
    best_fitness = float('inf')
    history = []
    no_improvement = 0

    for generation in range(max_generations):
        # 生成新一代参数种群
        solutions = es.ask()
        current_time = time.time() - start_time

        # 计算适应度
        fitness_values = [evaluate_fitness(
            model, x, train_dataloader,
            generation=generation,
            max_generations=max_generations) for x in solutions]

        # 更新 CMA-ES 内部状态
        es.tell(solutions, fitness_values)

        # 获取当前代最优
        current_best_fitness = min(fitness_values)
        current_best_params = solutions[np.argmin(fitness_values)]

        # 更新全局最优
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_params = current_best_params
            no_improvement = 0
            # 保存模型
            model.set_parameters(params_to_dict(best_params))
            torch.save(model.state_dict(), config.get_model_path())

            # 每代保存预测结果（前5代和后10代）
            if generation < 5 or generation > (max_generations - 10):
                save_generation_results(model, params_to_dict(best_params), generation)
        else:
            no_improvement += 1
            print(f'Count:{no_improvement}/{config.patience}: Best Fitness: {best_fitness:.6f}')

        # 记录结果
        elapsed_minutes = current_time / 60
        best_params_dict = params_to_dict(best_params) if best_params is not None else {}

        result = {
            'generation': generation + 1,
            'elapsed_minutes': elapsed_minutes,
            'current_best_fitness': current_best_fitness,
            'best_fitness': best_fitness,
            **best_params_dict
        }
        results.append(result)

        # 打印状态
        print(f"\nGeneration {generation + 1}/{max_generations} - Elapsed: {elapsed_minutes:.1f} min")
        print(f"Current Best Fitness: {current_best_fitness:.6f} | Global Best: {best_fitness:.6f}")
        print(f"Best Parameters: {best_params_dict}")

        # 早停检查
        if no_improvement >= config.patience:
            print(f"Early stopping at generation {generation + 1} (No improvement for {config.patience} generations)")
            break

    # 训练结束后保存最终结果
    save_final_results(model, results, params_to_dict(best_params))

    # 验证集评估
    val_fitness = evaluate_fitness(model, best_params, val_dataloader, use_regularization=False)
    print(f"Final Validation Fitness: {val_fitness:.6f}")

    return model, params_to_dict(best_params), best_fitness


# 保存每一代结果（辅助函数）
def save_generation_results(model, params_dict, generation):
    # 保存训练集预测
    train_pred = save_predictions(model, params_dict, 'train')

    # 保存验证集预测
    val_pred = save_predictions(model, params_dict, 'val')

    # 合并结果
    combined = pd.concat([train_pred, val_pred], ignore_index=False)

    # 保存路径
    '''
    save_path = f"./result/{config.zhanming}/gen_{generation + 1}_result{datetime.now().strftime('%m%d%H%M')}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path)
    print(f"Saved generation {generation + 1} predictions to: {save_path}")
'''

# 保存最终结果（基于ASA代码修改）
def save_final_results(model, results, best_params_dict):
    # 保存训练历史
    '''
    df = pd.DataFrame(results)
    history_path = f"./result/{config.zhanming}/cmaes_history_{config.zhanming}{config.time}.xlsx"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    df.to_excel(history_path, index=False)
    print(f"\nTraining history saved to: {history_path}")
    '''

    # 保存最佳参数预测
    final_preds = pd.concat([
        save_predictions(model, best_params_dict, 'train'),
        save_predictions(model, best_params_dict, 'val')
    ], ignore_index=False)

    # 添加时间戳
    # 获取保存路径
    datestr = datetime.now().strftime("%Y%m%d")
    final_preds_path = f"./result/{config.zhanming}/result{datestr}{config.time}.xlsx"

    # 使用xlsxwriter调整格式
    with pd.ExcelWriter(final_preds_path, engine='xlsxwriter') as writer:
        final_preds.to_excel(writer, sheet_name='Predictions')
        worksheet = writer.sheets['Predictions']
        worksheet.set_column('A:A', 20)  # 时间列宽度
        for i, col in enumerate(final_preds.columns[1:], start=1):
            worksheet.set_column(i, i, 12)

    print(f"Final predictions saved to: {final_preds_path}")


if __name__ == '__main__':
    model = TankModel().to(config.device)
    model, best_params, best_fitness = train(model, MAX_GENERATIONS)

    print("\n=== 训练完成 ===")
    print(f"最优参数: {best_params}")
    print(f"最低损失: {best_fitness:.6f}")
    print(f"结果保存在: ./result/{config.zhanming}/")