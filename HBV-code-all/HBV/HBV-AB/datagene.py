"""

作者： gaoshuai
日期： 2021年12月30日
"""

import config
import os
import datadeal
import pandas as pd
import re

timesteps = config.timesteps
leadtimes = config.leadtimes
time = config.time

class GeneData():
    def __init__(self):
        path = f'C:\\pycharm\\PUB\\data\\filter_data\\{config.zhanming}'
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(x.replace('event', '').replace('.csv', '')))
        print(files)
        set_train,set_val,set_test= datadeal.peak_divided(config.zhanming)
        raindistrain = None
        raindisval = None
        raindistest = None
        for i, file in enumerate(files):
            event = pd.read_csv(os.path.join(path, file), header=0, index_col=0)
            reshaped = datadeal.event_chuli(event, config.timesteps,file)
            if i in set_val:
                if raindisval is None:
                    raindisval = reshaped
                else:
                    raindisval = pd.concat([raindisval, reshaped])
                print(f'{file} is put into valset --- {i}')
            elif i in set_test:
                if raindistest is None:
                    raindistest = reshaped
                else:
                    raindistest = pd.concat([raindistest, reshaped])
                print(f'{file} is put into testset --- {i}')
            else:
                if raindistrain is None:
                    raindistrain = reshaped
                else:
                    raindistrain = pd.concat([raindistrain, reshaped])
                print(f'{file} is put into trainset --- {i}')
        # 在保存之前检查变量是否为 None
        if raindisval is not None:
            raindisval = raindisval.round(2)
        if raindistest is not None:
            raindistest = raindistest.round(2)
        if raindistrain is not None:
            raindistrain = raindistrain.round(2)

        target_data_foleder = f'..\\data_divided\\{config.zhanming}'
        os.makedirs(target_data_foleder, exist_ok=True)

        if raindisval is not None:
            raindisval.to_csv(f'{target_data_foleder}\\raindisval' + str(timesteps) + '_' + str(leadtimes) + '.csv')
        if raindistest is not None:
            raindistest.to_csv(f'{target_data_foleder}\\raindistest'+ str(timesteps) + '_' + str(leadtimes) + '.csv')
        if raindistrain is not None:
            raindistrain.to_csv(f'{target_data_foleder}\\raindistrain'+ str(timesteps) + '_' + str(leadtimes) + '.csv')

if __name__ == "__main__":
    gene = GeneData()
    gene
