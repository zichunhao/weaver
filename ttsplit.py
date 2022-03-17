"""
Splits sample files in the `source_dir` into training and testing data

Creates 'train' and 'test' directories in the `source_dir` and moves the files in them

"""

import os
from pathlib import Path
import random
import shutil
from math import ceil

source_dir = "/hwwtaggervol/training/ak15_Feb14/"
SPLIT_FRAC = 0.15  # fraction of data for testing

samples = os.listdir(source_dir)

for sample in samples:
    if sample == "test" or sample == "train":
        continue

    print(f"splitting {sample}")
    Path(f"{source_dir}/train/{sample}").mkdir(parents=True, exist_ok=True)
    Path(f"{source_dir}/test/{sample}").mkdir(parents=True, exist_ok=True)

    files = os.listdir(f"{source_dir}/{sample}")
    random.shuffle(files)
    split_index = ceil(len(files) * SPLIT_FRAC)

    for file in files[:split_index]:
        shutil.move(f"{source_dir}/{sample}/{file}", f"{source_dir}/test/{sample}/{file}")

    for file in files[split_index:]:
        shutil.move(f"{source_dir}/{sample}/{file}", f"{source_dir}/train/{sample}/{file}")
