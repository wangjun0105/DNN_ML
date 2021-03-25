import numpy as np
from ofdm import *
import os
from model import *
from config import *

gpu_list = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data1 = open('/home/wangjun/毕业设计/H.bin', 'rb')
H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
H = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]

Htest = H[256000:, :, :]
Htrain = H[:256000, :, :]


def ML(YY, H):
    mapping_table = np.array([-0.7071 - 0.7071j, -0.7071 + 0.7071j, 0.7071 - 0.7071j, 0.7071 + 0.7071j])
    x_hat = np.zeros([K, 2], dtype=complex)
    for i in range(K):
        metric = 10000000
        x_tmp = np.zeros(2, dtype=complex)
        for m in range(4):
            x_tmp[0] = mapping_table[m]
            est_y1 = YY[i] - H[0, i] * x_tmp[0]
            for n in range(4):
                x_tmp[1] = mapping_table[n]
                est_y2 = est_y1 - H[1, i] * x_tmp[1]
                metric_tmp = math.sqrt((abs(est_y2) * abs(est_y2)))

                if metric_tmp < metric:
                    metric = metric_tmp
                    x_hat[i] = x_tmp
    return x_hat


model00 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model00.load_state_dict(torch.load("DNN estimate para TX00.pth"))
model01 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model01.load_state_dict(torch.load("DNN estimate para TX01.pth"))
model10 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model10.load_state_dict(torch.load("DNN estimate para TX10.pth"))
model11 = NeuralNet(input_size, hidden_size1, output_size).to(device)
model11.load_state_dict(torch.load("DNN estimate para TX11.pth"))


def test():
    BER = np.zeros(8)

    SNRdb = np.linspace(8, 12, 8)

    for i in range(8):
        number = 0
        total_error = 0

        for k in range(1000):
            HH = Htest[k]
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))

            X = [bits0, bits1]



            H_tilde00, H_tilde01, H_tilde10, H_tilde11, Y_00, Y_01, Y_10, Y_11 = LS(X, HH, SNRdb[i], P, mode)
            YY = MIMO(X, HH, SNRdb[i], mode, P)

            test_input00 = Y_00
            test_input01 = Y_01
            test_input10 = Y_10
            test_input11 = Y_11

            test_input00 = torch.from_numpy(test_input00).float().to(device)
            test_input01 = torch.from_numpy(test_input01).float().to(device)
            test_input10 = torch.from_numpy(test_input10).float().to(device)
            test_input11 = torch.from_numpy(test_input11).float().to(device)

            with torch.no_grad():
                H_EST00 = model00(test_input00)
                H_EST01 = model01(test_input01)
                H_EST10 = model10(test_input10)
                H_EST11 = model11(test_input11)

                H_EST00 = H_EST00[0:K] + 1j * H_EST00[K:2 * K]
                H_EST01 = H_EST01[0:K] + 1j * H_EST01[K:2 * K]
                H_EST10 = H_EST10[0:K] + 1j * H_EST10[K:2 * K]
                H_EST11 = H_EST11[0:K] + 1j * H_EST11[K:2 * K]

                H_EST00 = H_EST00.cpu()
                H_EST01 = H_EST01.cpu()
                H_EST10 = H_EST10.cpu()
                H_EST11 = H_EST11.cpu()

                H_EST00 = H_EST00.detach().numpy()
                H_EST01 = H_EST01.detach().numpy()
                H_EST10 = H_EST10.detach().numpy()
                H_EST11 = H_EST11.detach().numpy()

                H_ESTR0 = np.concatenate((H_EST00, H_EST10))
                H_ESTR1 = np.concatenate((H_EST01, H_EST11))
                H_ESTR0 = np.reshape(H_ESTR0, [2, -1])
                H_ESTR1 = np.reshape(H_ESTR1, [2, -1])

                Y0 = YY[2 * K:3 * K] + 1j * YY[3 * K:4 * K]
                Y1 = YY[6 * K:7 * K] + 1j * YY[7 * K:8 * K]

                TX = ML(Y0, H_ESTR0)
                X0 = TX[:, 0]
                X1 = TX[:, 1]
                X0 = deModulation(X0)
                X1 = deModulation(X1)

                total_error += (X0 != X[0]).sum() + (X1 != X[1]).sum()
                number += 4 * K
        BER[i] = total_error / number

        np.savetxt("DNN+ML BER", BER, newline=" ")

test()

