import torch
from models.PINN_inter_model import PINN
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor

model = PINN()


rr_reg = TabPFNRegressor(device=torch.device('cuda'))
para_reg = TabPFNRegressor(device=torch.device('cuda'))
para_l = TabPFNRegressor(device=torch.device('cuda'))

pinn_data = pd.read_csv('csv/pinn_data.csv')

model.load_state_dict(torch.load('models/saved/PINN_model.pth'))

pinn_x = pinn_data.to_numpy()


validation = pd.read_csv('csv/pinn_data.csv').to_numpy(dtype=np.float64)



rr_reg.fit(np.log(validation[:,:7]), validation[:,8])

validation2 = pd.read_csv('csv/output.csv').to_numpy()



para_reg.fit(np.log(validation2[:,:7]), validation2[:,8])

para_l.fit(np.log(validation2[:,:7]), validation2[:,9])


data = pd.read_csv('csv/cma.csv')

data_np = torch.from_numpy(data.to_numpy(dtype=np.float32))

r_pre, l_pre = np.exp(model(torch.log(data_np[:,:6]),torch.log(data_np[:,6])).T.detach().numpy())

rr_p = rr_reg.predict(np.log(data.iloc[:,:7].to_numpy(dtype=np.float64)))
para_p = para_reg.predict(np.log(data.iloc[:,:7].to_numpy(dtype=np.float64)))
para_lp = para_l.predict(np.log(data.iloc[:,:7].to_numpy(dtype=np.float64)))


data["tabpfn"] =  rr_p
data["pinn"] = r_pre
data["ParaPFN"] = para_p
data["para_L"] = para_lp

data.to_csv("csv/validated.csv")

