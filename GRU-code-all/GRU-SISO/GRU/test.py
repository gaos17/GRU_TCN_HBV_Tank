'''
模型评估
'''
import argparse
import torch
from gruAB import Attenmodel
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
    model = Attenmodel().to(config.device)
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
        for dataset1 in ['train','val', 'test']:
            data_loader = get_dataloader(dataset1=dataset1)
            for idx,(time, input, flowbase, area,  target) in enumerate(data_loader):
                input = input.to(config.device)
                flowbase = flowbase.to(config.device)
                area = area.to(config.device)
                decoder_outputs = model(input, flowbase, area)
                decoder_outputs = decoder_outputs.cpu()
                target = target
                results = pd.concat(
                    [pd.DataFrame(time), pd.DataFrame(target.numpy()), pd.DataFrame(decoder_outputs.numpy())], axis=1)
                if idx == 0:
                    results_all = results
                else:
                    results_all = pd.concat([results_all, results], axis=0)
            results_all.columns = ['times', 'rainmean'] + ['real_' + str(i + 1) for i in range(config.leadtimes)] + ['pred_' + str(i + 1) for i in range(config.leadtimes)]
            results_all.sort_values(by='times', inplace=True)
            if dataset1 =='test':
                results_all.to_excel(paths["save_result"])
                print(results_all.head())
            nse, rmse, rr, pte = pr.evaluate(results_all)
            nse_rmse = pd.DataFrame([nse, rmse, rr, pte])
            if dataset1 =='train':
                NSE = nse_rmse
            else:
                NSE = pd.concat([NSE,nse_rmse])
        NSE.to_excel(paths["save_nse"])
        print(NSE)
        savepath = paths["save_resultpicture"]
        pr.plotrealpred(results_all, path=savepath, leadtime=config.plot_leadtime)


if __name__ == '__main__':
    eval()

