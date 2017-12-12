import pandas as pd
import os
import shutil
import numpy as np

label = 'unknown'
train_dir = os.path.join('..', 'input', 'test', 'audio')
dest_dir = os.path.join('..', 'input', 'submit_verify', label)
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

df = pd.read_csv(r'sub.csv')
fnames = df.fname[df.label == label]

fnames = np.random.choice(fnames, 1000)

for fname in fnames:
    shutil.copyfile(os.path.join(train_dir, fname), os.path.join(dest_dir, fname))