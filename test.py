import pandas as pd

df = pd.DataFrame(columns=[
"tCu","wCu","tLam","nLam","aln","tsu","freq"
])
df.to_csv("csv/cma.csv", index=False)