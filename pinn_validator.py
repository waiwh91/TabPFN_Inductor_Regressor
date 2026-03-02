import torch
from models.PINN_inter_model import PINN
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor

model = PINN()


rr_reg = TabPFNRegressor(device=torch.device('cuda'))
pinn_data = pd.read_csv('csv/pinn_data.csv')

model.load_state_dict(torch.load('models/saved/PINN_model.pth'))

pinn_x = pinn_data.to_numpy()


validation = pd.read_csv('csv/pinn_data.csv').to_numpy()
rr_reg.fit(validation[:,:7], validation[:,8])



data = pd.read_csv('csv/cma.csv')

data_np = torch.from_numpy(data.to_numpy(dtype=np.float32))

r_pre, l_pre = np.exp(model(torch.log(data_np[:,:6]),torch.log(data_np[:,6])).T.detach().numpy())

rr_p = rr_reg.predict(data.iloc[:,:7].to_numpy())

data["tabpfn"] =  rr_p
data["pinn"] = r_pre
data.to_csv("csv/validated.csv")