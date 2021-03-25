import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from ofdm import *
from config import *


def train_generator(Htrain):
    bits0 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    bits1 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    X = [bits0, bits1]
    H_tilde00, H_tilde01, H_tilde10, H_tilde11, Y_00, Y_01, Y_10, Y_11 = LS(X, Htrain, SNRdb, P, mode)
    data00 = Y_00
    data01 = Y_01
    data10 = Y_10
    data11 = Y_11
    H_label = np.fft.fft(Htrain, K)
    label00 = np.concatenate((np.real(H_label[0, :]), np.imag(H_label[0, :])))
    label01 = np.concatenate((np.real(H_label[1, :]), np.imag(H_label[1, :])))
    label10 = np.concatenate((np.real(H_label[2, :]), np.imag(H_label[2, :])))
    label11 = np.concatenate((np.real(H_label[3, :]), np.imag(H_label[3, :])))
    return data00, label00, data01, label01, data10, label10, data11, label11


def test_generator(Htest):
    bits0 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    bits1 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    X = [bits0, bits1]
    H_tilde00, H_tilde01, H_tilde10, H_tilde11, Y_00, Y_01, Y_10, Y_11 = LS(X, Htest, SNRdb, P, mode)
    data00 = Y_00  # , np.real(H_tilde00), np.imag(H_tilde00)))
    data01 = Y_01  # , np.real(H_tilde01), np.imag(H_tilde01)))
    data10 = Y_10  # , np.real(H_tilde10), np.imag(H_tilde10)))
    data11 = Y_11  # , np.real(H_tilde11), np.imag(H_tilde11)))
    H_label = np.fft.fft(Htest, K)
    label00 = np.concatenate((np.real(H_label[0, :]), np.imag(H_label[0, :])))
    label01 = np.concatenate((np.real(H_label[1, :]), np.imag(H_label[1, :])))
    label10 = np.concatenate((np.real(H_label[2, :]), np.imag(H_label[2, :])))
    label11 = np.concatenate((np.real(H_label[3, :]), np.imag(H_label[3, :])))

    return data00, label00, data01, label01, data10, label10, data11, label11


class train_Dataset(Dataset):

    def __init__(self, Htrain):
        self.H = Htrain

    def __getitem__(self, index):
        X00, Y00, X01, Y01, X10, Y10, X11, Y11, = train_generator(self.H[index])

        return X00, Y00, X01, Y01, X10, Y10, X11, Y11

    def __len__(self):
        return self.H.shape[0]


class test_Dataset(Dataset):

    def __init__(self, Htest):
        self.H = Htest

    def __getitem__(self, index):
        X00, Y00, X01, Y01, X10, Y10, X11, Y11, = test_generator(self.H[index])

        return X00, Y00, X01, Y01, X10, Y10, X11, Y11

    def __len__(self):
        return self.H.shape[0]


if __name__ == '__main__':
    print("test dataset00.py")