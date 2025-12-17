'''
CMA-ES
'''
import argparse
from dataset import get_dataloader
import torch
import config
import numpy as np
import cma
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

# CMA-ES 超参数
MAX_GENERATIONS = 200  # 最大迭代代数
INITIAL_SIGMA = 0.5    # 初始步长
POPULATION_SIZE = 40   # 种群大小
L1_LAMBDA = 0.001      # L1正则化系数
L2_LAMBDA = 0.008      # L2正则化系数
# K_FOLDS = 5          # 交叉验证折数


# 参数范围（每日单位）
PARAM_RANGES = {
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


# 将参数列表转换为字典
def params_to_dict(params):
    keys = list(PARAM_RANGES.keys())
    return {keys[i]: params[i] for i in range(len(keys))}


# 将字典转换为参数列表
def dict_to_params(params_dict):
    keys = list(PARAM_RANGES.keys())
    return [params_dict[key] for key in keys]

# 动态调整正则化系数
def get_dynamic_lambda(generation, max_generations):
    # 随着迭代进行逐渐减小正则化强度
    decay_factor = 1 - (generation / max_generations)
    return L1_LAMBDA * decay_factor, L2_LAMBDA * decay_factor

# 计算适应度（MSE）
def evaluate_fitness(model, params, dataloader,use_regularization=True, generation=None, max_generations=None):
    params_dict = params_to_dict(params)
    model.set_parameters(params_dict)  # 设置模型参数
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    for _, input, _, flowbase, area, target in dataloader:
        input = input.to(config.device)
        flowbase = flowbase.to(config.device)
        area = area.to(config.device)
        target = target.to(config.device)
        with torch.no_grad():
            output = model(config.ETP, input, config.initial_SM, config.initial_SUZ, config.initial_SLZ, area, flowbase)
            loss = criterion(output[:, 29:], target[:, 29:])
        total_loss += loss.item()
        mse_loss = total_loss / len(dataloader)

    # L1/L2正则化项
    if use_regularization:
      params_tensor = torch.tensor(params, dtype=torch.float32)
      # 动态调整部分
      if generation is not None and max_generations is not None:
          progress = generation / max_generations
          current_l1 = L1_LAMBDA * (1 - 0.8 * progress)  # L1保留20%基础值
          current_l2 = L2_LAMBDA * (1 - progress)  # L2完全线性衰减
      else:
          current_l1 = L1_LAMBDA
          current_l2 = L2_LAMBDA
      l1_reg = current_l1 * torch.norm(params_tensor, p=1)
      l2_reg = current_l2 * torch.norm(params_tensor, p=2)
      return mse_loss + l1_reg.item() + l2_reg.item()      # 总损失
    return mse_loss

def pretty_print_params(params_dict):
    print("\n=== 当前模型参数 ===")
    for param, value in params_dict.items():
        print(f"{value:.4f}")  # 控制小数点后4位
    print("==================\n")

# CMA-ES 算法训练
def train(model, max_generations=MAX_GENERATIONS):
    results = []
    start_time = time.time()
    train_dataloader = get_dataloader(dataset1='train')
    val_dataloader = get_dataloader(dataset1='val')

    # 初始化参数（在参数范围内随机采样）
    initial_params = np.array([np.random.uniform(low, high) for (low, high) in PARAM_RANGES.values()])

    # 参数边界（转换为 CMA-ES 需要的格式）
    bounds = np.array(list(PARAM_RANGES.values()))
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    # 初始化 CMA-ES 优化器
    es = cma.CMAEvolutionStrategy(
        initial_params,
        INITIAL_SIGMA,
        {'bounds': [lower_bounds, upper_bounds], 'popsize': POPULATION_SIZE}
    )

    best_params = None
    best_fitness = float('inf')
    history = []
    no_improvement = 0

    for generation in range(max_generations):
        # 生成新一代参数种群
        solutions = es.ask()

        # 记录时间和参数
        current_time = time.time() - start_time

        # 计算适应度
        fitness_values = [evaluate_fitness(
            model, x, train_dataloader,
            use_regularization=True,
            generation=generation,
            max_generations=max_generations) for x in solutions]

        # 更新 CMA-ES 内部状态
        es.tell(solutions, fitness_values)

        # 获取当前代最优
        current_best_fitness = min(fitness_values)
        current_best_params = solutions[np.argmin(fitness_values)]

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
            print(f"模型保存完成，参数形状已统一为1维")
        # 更新全局最优
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_params = current_best_params
            no_improvement = 0
            # 保存模型
            model.set_parameters(params_to_dict(best_params))
            # 在保存模型的地方调用
            save_model_fixed(model, config.get_model_path())
        else:
            no_improvement += 1
            print(f'Count:{no_improvement}/{config.patience}: Best Fitness: {best_fitness:.4f}')

        # 每代都记录（原条件 generation < 10 or (generation + 1) % 1 == 0 已移除）
        if best_params is not None:
            best_params_dict = params_to_dict(best_params)
            print(f"\n迭代 {generation + 1} 次用时: {current_time:.2f} 秒")
            print("当前全局最优参数:")
            pretty_print_params(best_params_dict)

            # 记录结果（现在记录的是全局最优参数）
            result = {
                'generation': generation + 1,
                'time': current_time,
                'FC': best_params_dict['FC'],
                'L': best_params_dict['L'],
                'LP': best_params_dict['LP'],
                'beta': best_params_dict['beta'],
                'K0': best_params_dict['K0'],
                'K1': best_params_dict['K1'],
                'Kd': best_params_dict['Kd'],
                'K2': best_params_dict['K2'],
                'gamma_a': best_params_dict['gamma_a'],
                'gamma_tau': best_params_dict['gamma_tau'],
                'current_best_fitness': current_best_fitness,
                'best_fitness': best_fitness
            }
            results.append(result)

        # 打印信息
        print(f"Generation {generation}: Current Best Fitness: {current_best_fitness:.4f}")
        print(f'Global Best Fitness: {best_fitness:.4f}')
        history.append({'generation': generation, 'fitness': current_best_fitness})

        # 提前终止
        if no_improvement > config.patience:
            print(f"Early stopping at generation {generation}")
            break

    # 保存结果
    #df = pd.DataFrame(results)
    #excel_path = config.get_result_path()
    #df.to_excel(excel_path, index=False)
    #print(f"\nResults saved to: {excel_path}")

    # 验证集评估
    val_fitness = evaluate_fitness(model, best_params, val_dataloader, use_regularization=False)
    print(f"Final Validation Fitness: {val_fitness:.4f}")

    return model, params_to_dict(best_params), best_fitness

if __name__ == '__main__':
    model = HBVModel().to(config.device)
    model, best_params, best_fitness = train(model, MAX_GENERATIONS)
    print(f"Final Best Parameters: {best_params}")
    print(f"Final Best Fitness: {best_fitness:.4f}")