import os
import shutil

with open(os.path.join('..', 'input', 'train', 'validation_list.txt'), 'r') as f:
    line = f.readline().strip()
    while line:
        dir, filename = line.split('/')
        dest_path = os.path.join('..', 'input', 'train', 'valid', dir)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        os.rename(os.path.join('..', 'input', 'train', 'audio', line), os.path.join(dest_path, filename))
        line = f.readline().strip()
