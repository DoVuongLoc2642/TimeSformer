
from sklearn.model_selection import train_test_split
import pandas as pd

paths = []
labels = []

with open("data/metadata/data.csv", "r", encoding="utf8") as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.replace("\n", "")
        paths.append(line.split(",")[0])
        labels.append(int(line.split(",")[1]))

paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.1)
    
df = pd.DataFrame()
df["path"] = paths_train
df["label"] = labels_train

df.to_csv("data/metadata/train.csv", index=False, header=False)

df = pd.DataFrame()
df["path"] = paths_val
df["label"] = labels_val

df.to_csv("data/metadata/val.csv", index=False, header=False)