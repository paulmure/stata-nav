import os
import random
import math
from shutil import copyfile
import pathlib

data_dir = 'stata_dataset'
output_dir = 'split_dataset'

for label in os.listdir(data_dir):
    input_dir = os.path.join(data_dir, label)
    train_output_dir = os.path.join(output_dir, 'train', label)
    pathlib.Path(train_output_dir).mkdir(parents=True)
    val_output_dir = os.path.join(output_dir, 'val', label)
    pathlib.Path(val_output_dir).mkdir(parents=True)

    images = os.listdir(input_dir)
    random.shuffle(images)

    split_idx = math.floor(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for ti in train_images:
        src = os.path.join(input_dir, ti)
        dst = os.path.join(train_output_dir, ti)
        copyfile(src, dst)
    for vi in train_images:
        src = os.path.join(input_dir, ti)
        dst = os.path.join(val_output_dir, ti)
        copyfile(src, dst)
