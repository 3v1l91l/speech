import os
import shutil

train_path = os.path.join('..', 'input', 'train')
# with open(os.path.join(train_path, 'validation_list.txt'), 'r') as f:
#     line = f.readline().strip()
#     while line:
#         directory, filename = line.split('/')
#         destination_path = os.path.join(train_path, 'valid', directory)
#         if not os.path.exists(destination_path):
#             os.makedirs(destination_path)
#         os.rename(os.path.join(train_path, 'audio', line), os.path.join(destination_path, filename))
#         line = f.readline().strip()


with open(os.path.join(train_path, 'testing_list.txt'), 'r') as f:
    line = f.readline().strip()
    while line:
        directory, filename = line.split('/')
        destination_path = os.path.join(train_path, 'test', directory)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        os.rename(os.path.join(train_path, 'audio', line), os.path.join(destination_path, filename))
        line = f.readline().strip()