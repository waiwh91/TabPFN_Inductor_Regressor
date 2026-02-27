import torch
import numpy as np
import pandas as pd
from models.PINN_inter_model import PINN


def test(X, targetR, targetL, f):
    model = PINN()
    model.load_state_dict(torch.load('models/saved/PINN_para_model.pth'))

    X = torch.from_numpy(X).float()

    r, l = np.exp(model(torch.log(X[:, :6]), torch.log(X[:, 6])).T.detach().numpy())
    print(r, l)
    designs = []
    freq = [f]
    print("===========Validating===========")
    for i in range(len(X)):
        temp = []
        if X[i,1] <= 300:
            if r[i] <= targetR * 1.1 and r[i] >= targetR * 0.9 and l[i] <= targetL * 1.1 and l[i] >= targetL * 0.9:
                print(i)
                temp = np.concatenate((X[i].reshape(-1), np.array([r[i], l[i]])))
                designs.append(temp)

    print("=========Validated Designs==========")
    print(designs)
    print(len(designs))
    designs = np.array(designs)

    output_df = pd.DataFrame(
        {"tCu": designs[:, 0], "wCu": designs[:, 1], "tLam": designs[:, 2], "nLam": designs[:, 3],
         "aln": designs[:, 4], "tsu": designs[:, 5], "freq": designs[:, 6],
         "Pre_R": designs[:, 7], "Pre_L": designs[:, 8]})

    output_df.to_csv("csv/pinn_validated_designs.csv", index=False)



