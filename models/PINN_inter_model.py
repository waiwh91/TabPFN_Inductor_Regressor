import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# from sklearn.preprocessing import MinMaxScaler


######### 划分数据集 split dataset into train/test

def split_data(data, proportion):
    x = data[:,:7]
    y = data[:,7:]

    x_train = x[:int(data.shape[0]*proportion), :]
    x_test = x[int(data.shape[0]*proportion):, :]

    y_train = y[:int(data.shape[0]*proportion), :]
    y_test = y[int(data.shape[0]*proportion):, :]

    return x_train, y_train, x_test, y_test

######### Define PINN models 定义模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 8)
        # self.ln1 = nn.LayerNorm(8)

        self.fc2 = nn.Linear(8+1, 32)

        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16, 6)
        self.fc5 = nn.Linear(6, 2)

    def forward(self, x, extra):
        # h = torch.tanh(self.fc1(x))
        h = nn.functional.silu(self.fc1(x))
        # h = self.ln1(h)

        extra = extra.unsqueeze(1)
        h = torch.cat([h, extra], dim=1)
        # h = torch.tanh(self.fc2(h))
        h = nn.functional.silu(self.fc2(h))
        h = self.ln2(h)
        # h = torch.tanh(self.fc3(h))
        h = nn.functional.silu(self.fc3(h))
        h = nn.functional.silu(self.fc4(h))
        out = self.fc5(h)
        # out = nn.Softmax(dim=1)(out)
        return out

def train(model, dataloader, epoches = 2000, alpha = 1.0, beta = 6.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    loss_q = nn.SmoothL1Loss()


    for epoch in range(epoches):
        epoch_loss = 0.0

        for batch_x, batch_f, batch_y, batch_q in dataloader:

            preds = model(batch_x, batch_f)

            loss_data = loss_fn(preds, batch_y)

            # R_pre = preds.detach().cpu().numpy()[:,0]
            # L_pre = preds.detach().cpu().numpy()[:,1]
            #
            #
            # f = batch_f.detach().cpu().numpy()

            R_pre = preds.detach()[:, 0]
            L_pre = preds.detach()[:, 1]

            f = batch_f.detach()


            omega = np.log(2) + np.log(torch.pi) + f
            Q_pre = omega + L_pre - R_pre


            # for i in range(len(Q_pre)):
            #     if Q_pre[i] > np.log(1000):
            #         Q_pre[i] = Q_pre[i] - np.log(1000)


            # loss_physics = loss_fn(torch.from_numpy(Q_pre).to(device), batch_q)
            loss_physics = loss_fn(Q_pre, batch_q)



            loss = alpha * loss_data + beta * loss_physics             ############ α 和 β 用来指定两种损失函数哪个比重大

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epoches}, Loss = {epoch_loss/len(dataloader):.6f}")

def test(model, data):
    x, f, Q_data, R_data, L_data = data



    R_pre, L_pre = model(x, f).T.detach().cpu().numpy()

    omega = np.log(2) + np.log(torch.pi) + f
    Q_pre = omega.cpu().numpy() + L_pre - R_pre

    # for i in range(len(Q_pre)):
    #     if Q_pre[i] > np.log(1000):
    #         Q_pre[i] = Q_pre[i] - np.log(1000)
    R_data = R_data.detach().cpu()
    L_data = L_data.detach().cpu()
    Q_data = Q_data.detach().cpu()
    f = f.detach().cpu()

    dataframe = pd.DataFrame({"Rpre":np.exp(R_pre), "Lpre":np.exp(L_pre), "Qpre":np.exp(Q_pre), "Rdata":np.exp(R_data), "Ldata":np.exp(L_data), "Qdata":np.exp(Q_data), "f":np.exp(f)})
    dataframe.to_csv("training_csv/PINN_output.csv", index=False)

    error_q_total = 0
    error_r_total = 0
    error_l_total = 0

    PREQ = Q_pre
    DATQ = Q_data.detach().cpu().numpy()

    PRER = R_pre
    DATR = R_data.detach().cpu().numpy()

    PREL = L_pre
    DATL = L_data.detach().cpu().numpy()

    DATF = f.detach().numpy()
    sumf = 0
    for i in range(len(PREQ)):


        sumf += 1

        error_q_total = error_q_total + (abs(PREQ[i] - DATQ[i]).item() / DATQ[i])
        error_r_total = error_r_total + (abs(PRER[i] - DATR[i]).item() / DATR[i])
        error_l_total = error_l_total + (abs(PREL[i] - DATL[i]).item() / DATL[i])


    print(f"RMSE of Q: {(error_q_total/sumf)}")
    print(f"RMSE of R: {(error_r_total/sumf)}")
    print(f"RMSE of L: {(error_l_total/sumf)}")

    return error_q_total/sumf, error_r_total/sumf, error_l_total/sumf

