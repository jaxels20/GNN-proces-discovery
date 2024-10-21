import numpy as np

# Load the npz file
data = np.load('datadir/sepsis/data.npz', allow_pickle=True)

# Inspect the names of the variables stored
print("Keys in the npz file:", data.files)

# If you want to print all arrays in the file
for key in data.files:
    print(f"{key}: {data[key]}")
