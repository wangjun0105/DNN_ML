from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import struct
import os
from ofdm import *
from dataset import *
from model import *
import time
from config import *


def worker_init_fn_seed(worker_id):
    seed = 10
    np.random.seed(seed)


gpu_list = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data1 = open('/home/wangjun/毕业设计/H.bin', 'rb')
H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
H = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]

Htest = H[256000:, :, :]
Htrain = H[:256000, :, :]


model00 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model01 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model10 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model11 = NeuralNet(input_size, hidden_size1, output_size).to(device)
num_workers = 2


def training():
    print("start training! ")
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer00 = optim.Adam(model00.parameters(), lr=learning_rate)
    optimizer01 = optim.Adam(model01.parameters(), lr=learning_rate)
    optimizer10 = optim.Adam(model10.parameters(), lr=learning_rate)
    optimizer11 = optim.Adam(model11.parameters(), lr=learning_rate)



    start_time = time.perf_counter()

    train_dataset = train_Dataset(Htrain)

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   worker_init_fn=worker_init_fn_seed)


    end_time = time.perf_counter()

    print("load data time%.4f" % (end_time - start_time))

    start_time = time.perf_counter()
    epoch_avg_loss = []
    for epoch in range(traing_epochs):
        print("========================================")
        print('Processing the', epoch + 1, 'th epoch')
        total_loss = 0.0
        batch_avg_loss = 0

        for i, data in enumerate(train_loader):

            input00, labels00, input01, labels01, input10, labels10, input11, labels11 = data
            input00 = input00.float().to(device)
            labels00 = labels00.float().to(device)
            input01 = input01.float().to(device)
            labels01 = labels01.float().to(device)
            input10 = input10.float().to(device)
            labels10 = labels10.float().to(device)
            input11 = input11.float().to(device)
            labels11 = labels11.float().to(device)

            output00 = model00(input00)
            output01 = model01(input01)
            output10 = model10(input10)
            output11 = model11(input11)

            loss00 = criterion(output00, labels00)
            loss01 = criterion(output01, labels01)
            loss10 = criterion(output10, labels10)
            loss11 = criterion(output11, labels11)

            optimizer00.zero_grad()
            loss00.backward()
            optimizer00.step()

            optimizer01.zero_grad()
            loss01.backward()
            optimizer01.step()

            optimizer10.zero_grad()
            loss10.backward()
            optimizer10.step()

            optimizer11.zero_grad()
            loss11.backward()
            optimizer11.step()

            total_loss += loss00.item()+ loss01.item()+loss10.item()+loss11.item()

            batch_avg_loss += loss11.item() + loss10.item() + loss01.item() + loss00.item()

            if (i + 1) % 10 == 0:
                print(
                    "Epoch: {}/{}, Batch {}/{}, MSE Loss {}".format(epoch + 1, traing_epochs, i + 1, len(train_loader),
                                                                    batch_avg_loss / 10))
                batch_avg_loss = 0

        epoch_avg_loss.append(total_loss / len(train_loader))
        print('第%d次循环，MSE on trainset %.4f' % (epoch + 1, epoch_avg_loss[epoch]))


        if epoch > 0:
            if (epoch_avg_loss[epoch - 1] - epoch_avg_loss[epoch] < 0.001):
                break
    print("optimization finished")

    end_time = time.perf_counter()
    print("train time%.4f" % (end_time - start_time))

    model00Save = "DNN estimate para TX00.pth"
    model01Save = "DNN estimate para TX01.pth"
    model10Save = "DNN estimate para TX10.pth"
    model11Save = "DNN estimate para TX11.pth"
    torch.save(model00.state_dict(), model00Save)
    torch.save(model01.state_dict(), model01Save)
    torch.save(model10.state_dict(), model10Save)
    torch.save(model11.state_dict(), model11Save)


def testing():
    total_loss = 0.0
    criterion = nn.MSELoss()

    test_dataset = test_Dataset(Htest)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  worker_init_fn=worker_init_fn_seed)
    model00.load_state_dict(torch.load("DNN estimate para TX00.pth"))
    model01.load_state_dict(torch.load("DNN estimate para TX01.pth"))
    model10.load_state_dict(torch.load("DNN estimate para TX10.pth"))
    model11.load_state_dict(torch.load("DNN estimate para TX11.pth"))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input00, labels00, input01, labels01, input10, labels10, input11, labels11 = data
            input00 = input00.float().to(device)
            labels00 = labels00.float().to(device)
            input01 = input01.float().to(device)
            labels01 = labels01.float().to(device)
            input10 = input10.float().to(device)
            labels10 = labels10.float().to(device)
            input11 = input11.float().to(device)
            labels11 = labels11.float().to(device)

            output00 = model00(input00)
            loss00 = criterion(output00, labels00)

            output01 = model01(input01)
            loss01 = criterion(output01, labels01)

            output10 = model10(input10)
            loss10 = criterion(output10, labels10)

            output11 = model11(input11)
            loss11 = criterion(output11, labels11)
            total_loss += loss00.item()+loss01.item()+loss10.item()+loss11.item()

    total_loss = total_loss/len(test_loader)


    print('MSE on testset %.4f' % total_loss)


training()
testing()

