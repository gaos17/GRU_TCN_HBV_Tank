'''
模型评估
'''
import argparse
import torch
from tankmodel0721 import TankModel
from dataset import get_dataloader
import config
import plotresults as pr
import pandas as pd
import numpy as np
import os
from flowgene import Flowgene

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--zhanming", type=str, required=True)
parser.add_argument("--time", type=int, required=True)
args = parser.parse_args()

# 动态更新配置
config.zhanming = args.zhanming
config.time = args.time
paths = config.get_save_paths()


def eval():
    model = TankModel().to(config.device)
    model.load_state_dict(torch.load(config.get_model_path()))
    # 下面代码用于输出模型参数
    jlxs = []
    hlxs = pd.DataFrame()
    for i in model.parameters():
        print(np.array(i.data.cpu()).reshape(-1))
        it = np.array(i.data.cpu()).reshape(-1)
        if len(it) == 1:
            jlxs.append(it)
        else:
            if len(hlxs) == 0:
                hlxs = pd.Series(it)
            else:
                hlxs = pd.concat([hlxs, pd.Series(it)], axis=0)
    cc1path = f'..\\factors_conv\\hlxs\\{config.zhanming}\\'
    cc2path = f'..\\factors_conv\\jlxs\\{config.zhanming}\\'
    os.makedirs(cc1path, exist_ok=True)
    os.makedirs(cc2path, exist_ok=True)
    cc1_path = os.path.join(cc1path,  str(config.time) + '.xlsx')
    hlxs.to_excel(cc1_path)
    cc2_path = os.path.join(cc2path, str(config.time) + '.xlsx')
    pd.DataFrame(jlxs).to_excel(cc2_path)

    with torch.no_grad():
        for dataset1 in ['train', 'val', 'test']:
            data_loader = get_dataloader(dataset1=dataset1)
            results_all = pd.DataFrame()

            for index, (time, packed_input, lengths, flowbase, area, target) in enumerate(data_loader):
                packed_input = packed_input.to(config.device)
                flowbase = flowbase.to(config.device)
                area = area.to(config.device)
                decoder_outputs = model(packed_input, config.initial_S, area, flowbase)
                decoder_outputs = decoder_outputs[:, 30:]
                decoder_outputs = decoder_outputs.cpu()
                target = target[:, 30:]
                target = target.numpy()

                # 遍历每个批次的数据
                all_time_data = []
                # 遍历批次中的每个事件
                for i in range(len(time)):  # len(time) 等于批次大小（batch_size）
                    time_array = time[i]  # 提取当前事件的时间序列
                    times = time_array[:, 0]  # 提取第一列：时间字符串
                    rainmeans = time_array[:, 1].astype(float)  # 提取第二列，并转换为 float
                    # 创建当前事件的时间和降雨 DataFrame
                    time_df = pd.DataFrame({
                        'times': pd.to_datetime(times[30:], errors='coerce'),  # 转换为 datetime
                        'rainmean': rainmeans[30:]
                    })
                    # 提取当前事件的目标值和预测值
                    target_df = pd.DataFrame(target[i, :].flatten(), columns=['real_1'])  # 当前事件的目标值
                    decoder_outputs_df = pd.DataFrame(decoder_outputs[i, :].flatten(), columns=['pred_1'])  # 当前事件的预测值
                    # 合并当前事件的结果
                    results = pd.concat([time_df, target_df, decoder_outputs_df], axis=1)
                    # 将当前事件的数据添加到列表中
                    all_time_data.append(results)
                    # 合并所有事件的数据
                results = pd.concat(all_time_data, axis=0, ignore_index=True)

                if index == 0:
                    results_all = results
                else:
                    results_all = pd.concat([results_all, results], axis=0)
                # 删除目标值和预测值均为 0 的多余行
            results_all = results_all[~((results_all['real_1'] == 0) & (results_all['pred_1'] == 0))]

            results_all.columns = ['times', 'rainmean'] + ['real_' + str(i + 1) for i in range(config.leadtimes)] + [
                'pred_' + str(i + 1) for i in range(config.leadtimes)]
            results_all.sort_values(by='times', inplace=True)

            # 按事件重新编号索引
            time_diff = results_all['times'].diff().dt.total_seconds()  # 计算时间差（秒）
            event_start = (time_diff > 3600) | (time_diff.isna())  # 判断事件起始点（时间差大于 1 小时或第一个时间点）
            event_ids = event_start.cumsum()  # 为每个事件分配唯一的事件编号
            # 遍历每个事件，重新编号索引
            results_all['new_index'] = results_all.groupby(event_ids).cumcount()
            # 替换原索引为新的索引
            results_all.set_index('new_index', inplace=True)
            results_all.index.name = None  # 清除索引名称

            if dataset1 == 'test':
                results_all.to_excel(paths["save_result"])
                print(results_all.head())
            nse, rmse, rr, pte = pr.evaluate(results_all)
            nse_rmse = pd.DataFrame([nse, rmse, rr, pte])
            if dataset1 == 'train':
                NSE = nse_rmse
            else:
                NSE = pd.concat([NSE, nse_rmse])
        NSE.to_excel(paths["save_nse"])
        print(NSE)
        savepath = paths["save_resultpicture"]
        pr.plotrealpred(results_all, path=savepath, leadtime=config.plot_leadtime)


if __name__ == '__main__':
    eval()

