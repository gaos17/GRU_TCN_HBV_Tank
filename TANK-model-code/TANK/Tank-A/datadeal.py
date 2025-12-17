'''
对数据进行滑动处理, 构造'30timestep+12leadtime'的格式,然后将处理好的数据传给dataset.py
'''


import numpy as np
import pandas as pd
import config
import datetime



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def hua_chuang(data, timesteps):
    df = data
    df1 = pd.DataFrame(columns = df.columns)
    for i in range(len(df)-timesteps+1):
        if pd.to_datetime(df.iloc[i+timesteps-1,0]) == pd.to_datetime(df.iloc[i,0]) + (timesteps-1)*datetime.timedelta(seconds=3600):
            df1 = pd.concat([df1, df.iloc[i:i+timesteps]])
    return df1

def event_chuli(event,timesteps,filename):
    #将流量数据dis换到最后一列
    file_name = filename
    print(f"正在处理{filename}")#此参数未应用
    reframed = series_to_supervised(pd.DataFrame(event.dis), 0, 1+config.leadtimes)
    event = pd.merge(event,reframed,left_index=True,right_index=True,how='right')
    meanrain = event.rain
    event.drop(['dis'],axis=1,inplace= True)
    event.insert(1, 'rainmean', meanrain)
    #如果要改leadtimes, 下面这一行必须要改
    event.columns = ['times','rainmean','rain','baseflow','area','dis']+['dis'+str(i+1) for i in range(config.leadtimes)]#,'dis7','dis8','dis9','dis10','dis11','dis12']
    event.index = np.arange(len(event))
    reshaped = hua_chuang(event,timesteps)
    # reshaped.drop(['times'], axis=1,inplace= True)
    return reshaped

def count_divided(files):#此函数用于事件划分
    file_count = len(files)
    part_size = file_count // 5
    remainder = file_count % 5
    sizes = [part_size * 3, part_size, part_size]
    for i in range(remainder):
        if i == 0:
            sizes[0] += 1
        elif i == 1:
            sizes[1] += 1
        elif i == 2:
            sizes[2] += 1
        elif i == 3:
            sizes[0] += 1
        else:
            sizes[1] += 1
    start_index = 0
    index_list = []
    for i, size in enumerate(sizes):
        end_index = start_index + size - 1
        index_list.extend([start_index, end_index])
        start_index = end_index + 1
    # 赋值给指定变量
    return index_list


def peak_divided(zhanming):  # 此函数用于按洪峰划分
    # 打开并读取 xlsx 文件为 DataFrame
    file_path = f'C:\\pycharm\\PUB\\data\\filter_data\\附录\\{zhanming}.xlsx'
    df = pd.read_excel(file_path)

    # 按倒数第四列的数值大小进行从大到小的排序
    df = df.sort_values(by=df.columns[-4], ascending=False)

    # 创建空列表
    set_train = []
    set_val = []
    set_test = []

    # 获取所有列名
    cols = list(df.columns)

    # 根据索引选择列
    selected_cols = [cols[0], cols[4], cols[-1]]
    df_new = df[selected_cols]

    # 滑动窗格处理
    i = 0
    while i < len(df_new):
        window1 = df_new.iloc[i:i + 5]
        if i >= 1:
            window1 = window1.sort_values(by=df.columns[-1], ascending=True)
        # 获取第一列的数据
        windows = window1.iloc[:, 0].tolist()
        window = [j - 1 for j in windows]
        print(f"Window starting at index {i}: {window}")

        if len(window) == 5:
            if i in [0, 25, 50]:
                set_train.extend([window[0], window[3], window[4]])
                set_val.append(window[1])
                set_test.append(window[2])
            if i in [5, 30, 55]:
                set_train.extend([window[0], window[1], window[4]])
                set_val.append(window[2])
                set_test.append(window[3])
            if i in [10, 35, 60]:
                set_train.extend([window[0], window[1], window[2]])
                set_val.append(window[3])
                set_test.append(window[4])
            if i in [15, 40, 65]:
                set_train.extend([window[1], window[2], window[3]])
                set_val.append(window[4])
                set_test.append(window[0])
            if i in [20, 45, 70]:
                set_train.extend([window[2], window[3], window[4]])
                set_val.append(window[0])
                set_test.append(window[1])
        elif len(window) == 4:
            set_train.extend([window[3], window[0]])
            set_val.append(window[1])
            set_test.append(window[2])
        elif len(window) == 3:
            set_train.extend([window[1], window[2], window[0]])
        elif len(window) == 2:
            set_train.extend([window[0], window[1]])
        elif len(window) == 1:
            set_train.append(window[0])
        i += 5

    print(set_train, set_val, set_test)
    return set_train, set_val, set_test


if __name__ == '__main__':

    event = pd.read_csv(f'C:\\pycharm\\PUB\\data\\filter_data\\{config.zhanming}\\event0.csv', index_col=0)
    print(event.head())
    event_reshaped = event_chuli(event, config.timesteps,0)
    #event_reshped:['times','rainmean','shuixi','quanshang','hucun', 'yutan','dis', 'dis1','dis2','dis3','dis4','dis5','dis6']
    print(event_reshaped.head(100))