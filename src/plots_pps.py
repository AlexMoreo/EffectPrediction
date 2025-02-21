import os
from os.path import join
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

dataset = 'activity'
n_classes = 3
result_path = f'../results/random_split/min1000_size500/{dataset}/{n_classes}_classes/*.csv'

reports = []
for csv_path in glob(result_path):
    df = pd.read_csv(csv_path, index_col=0)
    reports.append(df)

df = pd.concat(reports)

print(df)

df["method_features"] = df["method"] + " (" + df["features"] + ")"
df["method_features"] = df["method_features"].str.replace(r"MLPE .*", "MLPE", regex=True)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="tr_size", y="nmd", hue="method_features", marker="o", palette="tab10")

plt.xlabel("Training Size")
plt.ylabel("NMD Error")
plt.title(f"{dataset} ({n_classes} classes)")

plt.legend(title="Method (Features)")

outpath = '../fig/random_split/'
os.makedirs(outpath, exist_ok=True)
plt.savefig(join(outpath,f'{dataset}_{n_classes}classes.png'))
# plt.show()
