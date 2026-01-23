import itertools
import numpy as np
import pandas as pd
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
from tabpfn import TabPFNRegressor

tcu_range = [10,20]
wcu_range = [200,250]
tlam_range = [100, 350]
nlam_range = [8, 16]
aln_range = [15, 20]
tsu8_range= [2, 6]
freq = [1, 25.75, 50.5, 75.25, 100]


para_range = [tcu_range, wcu_range, tlam_range, nlam_range, aln_range, tsu8_range]

tcu_lst = []
wcu_lst = []
tlam_lst = []
nlam_lst = []
aln_lst = []
tsu8_lst = []

for i in np.arange(tcu_range[0], tcu_range[1],2):
    tcu_lst.append(i)


for i in np.arange(wcu_range[0], wcu_range[1],10):
    wcu_lst.append(i)

for i in np.arange(tlam_range[0], tlam_range[1],20):
    tlam_lst.append(i)

for i in np.arange(nlam_range[0], nlam_range[1],1):
    nlam_lst.append(i)

for i in np.arange(aln_range[0], aln_range[1],1):
    aln_lst.append(i)

for i in np.arange(tsu8_range[0], tsu8_range[1],1):
    tsu8_lst.append(i)

x_pre = list(itertools.product(tcu_lst, wcu_lst, tlam_lst, nlam_lst, aln_lst, tsu8_lst, freq))

print(len(x_pre))


train_data = pd.read_csv('../ML_Inductor_QLR_Predictor/training_csv/interpolation_data.csv').to_numpy()

X_train = train_data[:,:7]
y_train_R = train_data[:,8]
y_train_L = train_data[:,9]

R_reg = TabPFNRegressor(device = "cuda")
R_regressor = RandomForestTabPFNRegressor(tabpfn=R_reg)
L_reg = TabPFNRegressor(device = "cuda")
L_regressor = RandomForestTabPFNRegressor(tabpfn=L_reg)
R_regressor.fit(X_train, y_train_R)
L_regressor.fit(X_train, y_train_L)

pre_L = []
pre_R = []

for i in range(len(x_pre)//5000):
    print(i)
    pre_L.append(L_regressor.predict(x_pre[i*5000:(i + 1)* 5000]))
    pre_R.append(R_regressor.predict(x_pre[i*5000:(i + 1)* 5000]))

    print(len(pre_L), len(pre_R))

    for j in range(0, 4999):
        if pre_R[i][j] <= 260 and pre_R[i][j] >= 220:
            if pre_L[i][j] <= 5.7 and pre_L[i][j] >= 5.1:
                print(x_pre[i*5000 + j], pre_L[i][j], pre_R[i][j])