import torch
from models.PINN_inter_model import PINN
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor

model = PINN()

R_reg = TabPFNRegressor(device=torch.device('cuda'))
rr_reg = TabPFNRegressor(device=torch.device('cuda'))
pinn_data = pd.read_csv('csv/pinn_data.csv')

model.load_state_dict(torch.load('models/saved/PINN_inter_model.pth'))

pinn_x = pinn_data.to_numpy()
R_reg.fit(pinn_x[:,:7], pinn_x[:,8])

validation = pd.read_csv('csv/output.csv').to_numpy()
rr_reg.fit(validation[:,:7], validation[:,8])



data = pd.read_csv('csv/pinn_validated_designs.csv')

data_np = torch.from_numpy(data.to_numpy(dtype=np.float32))

r_pre, l_pre = np.exp(model(torch.log(data_np[:,:6]),torch.log(data_np[:,6])).T.detach().numpy())

R_predictions = R_reg.predict(data.iloc[:,:7].to_numpy())
rr_p = rr_reg.predict(data.iloc[:,:7].to_numpy())

data["r_pre"] = R_predictions
data["r_pre_reg"] =  rr_p
data["pinn_r"] = r_pre
data.to_csv("csv/PINN_validated.csv")