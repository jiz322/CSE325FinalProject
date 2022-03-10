import numpy as np
from tqdm import tqdm

with open("amazon_total.txt") as file:
    lines = file.readlines()

lines = np.array(lines)