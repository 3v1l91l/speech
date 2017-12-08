import pandas as pd
import os
import shutil

label = 'unknown'
# train_dir = os.path.join('..', 'input', 'test', 'audio')
train_dir = r'C:\code\speech\input\test\audio'
# dest_dir = os.path.join('..', 'input', 'submit_verify', label)
dest_dir = r'C:\code\speech\input\test\submit_verify\label'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

df = pd.read_csv(r'C:\code\speech\src\sub.csv')
fnames = df.fname[df.label == label]

for fname in fnames:
    shutil.copyfile(os.path.join(train_dir, fname), os.path.join(dest_dir, fname))