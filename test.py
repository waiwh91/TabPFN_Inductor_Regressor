import torch
from models.PINN_inter_model import PINN
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor

model = PINN()


rr_reg = TabPFNRegressor(device=torch.device('cuda'))
para_reg = TabPFNRegressor(device=torch.device('cuda'))
para_l = TabPFNRegressor(device=torch.device('cuda'))



validation = pd.read_csv('csv/pinn_data.csv').to_numpy()
rr_reg.fit(validation[:,:7], validation[:,8])

validation2 = pd.read_csv('csv/output.csv').to_numpy()
para_reg.fit(validation2[:,:7], validation2[:,8])


para_l.fit(validation2[:,:7], validation2[:,9])


data = np.log(pd.read_csv('csv/cma.csv').to_numpy())
data = np.exp(data)


rr_p = rr_reg.predict(data[:,:7])
para_p = para_reg.predict(data[:,:7])
para_lp = para_l.predict(data[:,:7])


print(para_p)
