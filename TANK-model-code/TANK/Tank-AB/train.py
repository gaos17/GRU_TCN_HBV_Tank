'''
train程序
'''
import argparse
import os
from dataset import get_dataloader
import torch
import config
import numpy as np
import plotresults as pr
from tank import TankModel

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--zhanming", type=str, required=True)
parser.add_argument("--time", type=int, required=True)
args = parser.parse_args()

# 动态更新配置
config.zhanming = args.zhanming
config.time = args.time

model = TankModel().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

def move_data_to_device(data_list, device):
    return [data.to(device) for data in data_list]

def train(model, epochs, patience = 10, ):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(epochs):

        model.train()

        train_dataloader = get_dataloader(dataset1='train')

        for index, (times, packed_input, lengths, flowbase, area, target) in enumerate(train_dataloader):
            data_list = [packed_input, flowbase, area, target]
            packed_input, flowbase, area, target = move_data_to_device(data_list, config.device)
            initial_S = config.initial_S.to(config.device)
            optimizer.zero_grad()
            decoder_outputs = model(packed_input, initial_S, area, flowbase)
            loss = criterion(decoder_outputs[:, 30:], target[:, 30:])
            loss.backward()
            # 在训练循环中添加（每次参数更新后）
            optimizer.step()
            model.apply_param_constraints()  # 应用参数约束
            train_losses.append(loss.item())

        model.eval()
        val_dataloader = get_dataloader(dataset1='val')
        with torch.no_grad():
            for index, (times, packed_input, lengths, flowbase, area, target) in enumerate(val_dataloader):
                data_list = [packed_input, flowbase, area, target]
                packed_input, flowbase, area, target = move_data_to_device(data_list, config.device)
                initial_S = config.initial_S.to(config.device)
                output = model(packed_input, initial_S, area, flowbase)
                loss = criterion(output[:, 30:], target[:, 30:])
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {valid_loss:.3f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        if epoch == 0:
            min_loss = valid_loss
            es = 0
        if valid_loss < min_loss:
            min_loss = valid_loss
            es = 0
            torch.save(model.state_dict(), config.get_model_path())
            torch.save(optimizer.state_dict(), config.get_optimizer_path())
            print("saving, model")
        else:
            es += 1
            print("Counter {} of 50".format(es))

            if es > patience:
                print("Early stopping with min_loss: ", min_loss)
                break

    return model, avg_train_losses, avg_valid_losses

if __name__ == '__main__':
    model, train_loss, val_loss = train(model, 1000, config.patience)
    pr.plotloss(train_loss, val_loss)



