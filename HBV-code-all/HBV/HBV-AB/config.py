'''
参数文件
'''
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import copy
# 所有需要运行的站点列表
ZHANMING_LIST = ['chenda','yutan','longmen','yangjiafang','yongding' ,'maiyuan','guanyinqiao']  # 自行修改站点列表

# 所有需要运行的time参数列表
TIME_VALUES = range(10)  # 自行修改time参数列表
# 动态参数（运行时会被覆盖）
zhanming = 'default_station'
time = 0

flowgenemethod = "convgelu" # "convrelu" "convsigmoid"
flowconfmethod = "unitattention" # "selfattention" "unitattention" "gru1"


#通用的一些参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
timesteps =30
leadtimes = 1
site = 1
# area = torch.tensor([372/3.6,137/3.6,180/3.6,80/3.6]).to(device)
target_dim = 1
hidden_size = 16
num_layer =1
conv1d_out_channel = 1
attn_method ='general'
FC = 100.0
L = 0.5
LP = 0.3
beta = 1.5
K0 = 0.1
K1 = 0.05
Kd = 0.02
K2 = 0.01
ETP = torch.tensor([0.0])
initial_SM = torch.tensor(50.0).to(device) # 初始土壤湿度
initial_SUZ = torch.tensor(20.0).to(device) # 初始上层土壤含水量
initial_SLZ = torch.tensor(30.0).to(device)# 初始下层土壤含水量


patience = 30

# dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#结果处理相关参数---test.py

plot_leadtime = 1

#def get_result_path():
    #datestr = datetime.strftime(datetime.now(), "%Y%m%d")
    #return f"./canshu/{zhanming}/canshu{datestr}{time}.xlsx"
def get_model_path():
    return f"./model/{zhanming}/seq2seq_{time}_attention.model"

def get_optimizer_path():
    return f"./model/{zhanming}/optimizer{time}_attention_model"

def get_save_paths():
    datestr = datetime.strftime(datetime.now(), "%Y%m%d")
    return {
        "save_nse": f"../../../result/{zhanming}/nse_rmse{datestr}{time}.xlsx",
        "save_result": f"../../../result/{zhanming}/result{datestr}{time}.xlsx",
        "save_plotloss": f'../../../picture/{zhanming}/loss_plot{datestr}{time}.png',
        "save_resultpicture": f"../../../hydrograph/{zhanming}/hydrograph{datestr}{time}.png",
    }

def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

