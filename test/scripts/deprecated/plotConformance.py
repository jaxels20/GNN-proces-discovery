import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', help='data directory', type=str)
args = parser.parse_args()

with open(f'{args.data_directory}/predictions/data_conformance_best.csv', 'r') as file:
  reader = csv.reader(file)
  table = list(reader)

X = ['fScore', 'Fitness', 'Precision', 'Simplicity', 'Generalization', 'MetricsAverageWeight']
header = [v.lower() for v in table[0]]
X_indices = [header.index(x.lower()) for x in X]

data = {}
for row in table[1:]:
  data[row[0]] = [float(row[i]) for i in X_indices]

data_sorted = {}
for key in sorted(data.keys()):
  data_sorted[key] = data[key]

print(data_sorted)

# plt.style.use('seaborn-pastel')
plt.style.use('ggplot')
# plt.style.use('fivethirtyeight')

df = pd.DataFrame(data_sorted, index=X)
df.plot.bar(rot=0)
plt.title(f'{args.data_directory}')

plt.show()
