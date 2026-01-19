from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn import TabPFNRegressor
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import pandas as pd
import numpy as np

train_data = pd.read_csv('output.csv').to_numpy()
test_data = pd.read_csv("../ML_Inductor_QLR_Predictor/training_csv/pinn_data.csv").to_numpy()

train_data[:,7: ] = np.log(train_data[:,7: ])
test_data[:,7: ] = np.log(test_data[:,7: ])

print(train_data.shape)
print(test_data.shape)


#################### tcu

tcu_train_y = train_data[:,0]
tcu_regressor = TabPFNRegressor(device="cuda")


tcu_train_x = train_data[:, 1:]

tcu_regressor.fit(tcu_train_x, tcu_train_y)

tcu_test_x = test_data[:,1:]
tcu_test_y = test_data[:,0]

tcu_prediction = tcu_regressor.predict(tcu_test_x)
tcu_mpe = mean_absolute_percentage_error(tcu_prediction, tcu_test_y)
tcu_r2 = r2_score(tcu_prediction, tcu_test_y)
print("tCu Predict:")
print("Mean percentage Error (MPE) for tcu:", tcu_mpe)
print("R² Score:", tcu_r2)

######################## wCu

wcu_train_y = train_data[:,1]
wcu_regressor = TabPFNRegressor(device="cuda")

wcu_train_x = np.concatenate((train_data[:,0:1], train_data[:,2:]), axis=1)

wcu_regressor.fit(wcu_train_x, wcu_train_y)

wcu_test_x = np.concatenate((test_data[:,0:1], test_data[:,2:]), axis=1)
wcu_test_y = test_data[:,1]

wcu_prediction = wcu_regressor.predict(wcu_test_x)
wcu_mpe = mean_absolute_percentage_error(wcu_prediction, wcu_test_y)
wcu_r2 = r2_score(wcu_prediction, wcu_test_y)
print("wCu Predict:")
print("Mean percentage Error (MPE) for wcu:", wcu_mpe)
print("R² Score:", wcu_r2)


######################## tLam

tlam_train_y = train_data[:,2]
tlam_regressor = TabPFNRegressor(device="cuda")

tlam_train_x = np.concatenate((train_data[:,0:2], train_data[:,3:]), axis=1)

tlam_regressor.fit(tlam_train_x, tlam_train_y)

tlam_test_x = np.concatenate((test_data[:,0:2], test_data[:,3:]), axis=1)
tlam_test_y = test_data[:,2]

tlam_prediction = tlam_regressor.predict(tlam_test_x, output_type="median")
tlam_mpe = mean_absolute_percentage_error(tlam_prediction, tlam_test_y)
tlam_r2 = r2_score(tlam_prediction, tlam_test_y)
print("tlam Predict:")
print("Mean percentage Error (MPE) for tlam:", tlam_mpe)
print("R² Score:", tlam_r2)


######################## nLam

nlam_train_y = train_data[:,3]
nlam_regressor = TabPFNRegressor(device="cuda")

nlam_train_x = np.concatenate((train_data[:,0:3], train_data[:,4:]), axis=1)

nlam_regressor.fit(nlam_train_x, nlam_train_y)

nlam_test_x = np.concatenate((test_data[:,0:3], test_data[:,4:]), axis=1)
nlam_test_y = test_data[:,3]

nlam_prediction = nlam_regressor.predict(nlam_test_x, output_type="median")
nlam_mpe = mean_absolute_percentage_error(nlam_prediction, nlam_test_y)
nlam_r2 = r2_score(nlam_prediction, nlam_test_y)
print("nlam Predict:")
print("Mean percentage Error (MPE) for nlam:", nlam_mpe)
print("R² Score:", nlam_r2)


######################## aln

aln_train_y = train_data[:,4]
aln_regressor = TabPFNRegressor(device="cuda")

aln_train_x = np.concatenate((train_data[:,0:4], train_data[:,5:]), axis=1)

aln_regressor.fit(aln_train_x, aln_train_y)

aln_test_x = np.concatenate((test_data[:,0:4], test_data[:,5:]), axis=1)
aln_test_y = test_data[:,4]

aln_prediction = aln_regressor.predict(aln_test_x)
aln_mpe = mean_absolute_percentage_error(aln_prediction, aln_test_y)
aln_r2 = r2_score(aln_prediction, aln_test_y)
print("aln Predict:")
print("Mean percentage Error (MPE) for aln:", aln_mpe)
print("R² Score:", aln_r2)


######################## tsu8

tsu8_train_y = train_data[:,5]
tsu8_regressor = TabPFNRegressor(device="cuda")

tsu8_train_x = np.concatenate((train_data[:,0:5], train_data[:,6:]), axis=1)

tsu8_regressor.fit(tsu8_train_x, tsu8_train_y)

tsu8_test_x = np.concatenate((test_data[:,0:5], test_data[:,6:]), axis=1)
tsu8_test_y = test_data[:,5]

tsu8_prediction = tsu8_regressor.predict(tsu8_test_x)

tsu8_mpe = mean_absolute_percentage_error(tsu8_prediction, tsu8_test_y)
tsu8_r2 = r2_score(tsu8_prediction, tsu8_test_y)
print("tsu8 Predict:")
print("Mean percentage Error (MPE) for tsu8:", tsu8_mpe)
print("R² Score:", tsu8_r2)


train_data[:,7:] = np.exp(train_data[:,7:])

output_df = pd.DataFrame(
        {"tCu": tcu_prediction, "wCu": wcu_prediction, "tLam": tlam_prediction, "nLam": nlam_prediction,
             "aln": aln_prediction, "tsu": tsu8_prediction, "freq": train_data[:,6], "Pre_Q": train_data[:,7],
             "Pre_R": train_data[:,8], "Pre_L": train_data[:,9]})



output_df.to_csv("paraout.csv", index=False)