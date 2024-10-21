import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from os import listdir
from os.path import isdir, join
import string
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', help='data directory', type=str)
parser.add_argument('-e', '--export', help='export', action='store_true')
parser.add_argument('-m', '--metrics', help='Metrics', nargs='+', action='append')
args = parser.parse_args()

path = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data/'

dataset_names = sorted([f for f in listdir(path) if isdir(join(path, f)) and f[0] != '.'])
dataset_names = dataset_names[-2:] + dataset_names[:-2]
print(dataset_names)

X = ['fscore', 'fitness', 'precision', 'simplicity', 'generalization', 'metricsAverageWeight',
     'fscore_alignments', 'fitness_alignments', 'precision_alignments', 'entropia_recall', 'entropia_precision',
     'entropia_partial_recall', 'entropia_partial_precision']

X_labels = {
  'fscore': 'F-score_',
  'fitness': 'Fitness_',
  'precision': 'Precision_',
  'simplicity': 'Simplicity',
  'generalization': 'Generalization',
  'metricsAverageWeight': 'metricsAverageWeight',
  'fscore_alignments': 'F-score\n(alignment)',
  'fitness_alignments': 'Fitness\n(alignment)',
  'precision_alignments': 'Precision\n(alignment)',
  'entropia_recall': 'Fitness\n(entropy)',
  'entropia_precision': 'Precision\n(entropy)',
  'entropia_fscore': 'F-Score\n(entropy)',
  'entropia_partial_recall': 'Fitness_pe',
  'entropia_partial_precision': 'Precision_pe'

}

# X = ['MetricsAverageWeight', 'Simplicity']

X = args.metrics[0]

methods = ['split_reduced', 'gcn', 'inductive_reduced', 'heuristics_reduced', 'ilp_reduced']

def find_method(filename):
  for method in methods:
    if method == 'gcn' and 'gcn' in filename:
      return method
    elif filename == f'data_{method}_cc_temp2.txt':
      return method

def process_dict(data):
  new_one = {}
  for key, value in data.items():
    if value is None or value == 'None' or value == 'nan':
      value = 0
    new_one[key.rstrip()] = float(value)
  return new_one

datasets = {'Average': {}}
for dataset in dataset_names:
  datasets[dataset] = {}
  path = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data/{dataset}/results/'
  conformance_files = sorted([f for f in listdir(path) if f[-9:] == 'temp2.txt'])
  for conformance_file in conformance_files:
    method = find_method(conformance_file)
    print(method)
    if method is not None:
      with open(f'{path}{conformance_file}', 'r') as file:
        data = process_dict(json.load(file)['full'])
        try:
          data['entropia_fscore'] = (2 * data.get('entropia_recall', 0) * data.get('entropia_precision', 0)) / (data.get('entropia_recall', 0) + data.get('entropia_precision', 0))
        except ZeroDivisionError:
          data['entropia_fscore'] = 0
        datasets[dataset][method] = [data.get(key, 0) for key in X]
        datasets['Average'].setdefault(method, []).append([data.get(key, 0) for key in X])

avg = datasets['Average']
del datasets['Average']
datasets['Average'] = avg

from temp_table import table
table(datasets)
print(fdsa)

# print(datasets)
# print(a)
#
# for dataset_name in dataset_names:
#   datasets[dataset_name] = {}
#   with open(f'/home/dominique/TUe/thesis/git_data/evaluation_data/{dataset_name}/predictions/data_conformance_best.csv', 'r') as file:
#     reader = csv.reader(file)
#     table = list(reader)
#     data = {}
#     header = [v.lower() for v in table[0]]
#     X_indices = [header.index(x.lower()) for x in X]
#     for row in table[1:]:
#       data[row[0]] = [float(row[i]) for i in X_indices]
#     data_sorted = {}
#     for key in sorted(data.keys()):
#       datasets[dataset_name][key] = data[key]
#
# print(datasets)
# print(A)

plt.style.use('ggplot')
# fig, ax = plt.subplots(figsize=(13, 7))

labels = {
  'Average': 'Average\n',
  'BPI_2012_A': "'12\nA_ prefixed",
  'BPI_2012_O': "'12\nO_ prefixed",
  'BPI_2017_A': "'17\nA_ prefixed",
  'BPI_2017_O':"'17\nO_ prefixed",
  'BPI_2020_Domestic_declarations': "'20\nDomestic\nDeclarations",
  'BPI_2020_International_declarations': "'20\nInternational\nDeclarations",
  'BPI_2020_Permit_log': "'20\nPermit Log\n",
  'BPI_2020_Prepaid_travel_cost': "'20\nPrepaid\nTravel Cost",
  'BPI_2020_Request_for_payment': "'20\nRequest for\nPayment",
  'road_traffic_fine': "Road Traffic\nFine",
  'sepsis': "Sepsis",
  'ilp_reduced': 'ILP Miner',
  'split': 'Split Miner',
  'split_reduced': 'Split Miner',
  'heuristics_reduced': 'Heuristics Miner',
  'inductive_reduced': 'Inductive Miner',
  'alpha': 'Alpha Miner',
}

markers = ['o', 'v', '1', 's', 'P', 'H', '^', '*', 'p', 'X', 'd']
# markers = [f'${i}$' for i, l in enumerate(string.ascii_lowercase[:11])]
markers = ['o']*20

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_algo_label(algorithm):
  if 'gcn' in algorithm:
    return 'Our approach'
  else:
    return labels[algorithm]

def get_color(algorithm):
  return {
    'Our approach': colors[0],
    'ILP Miner': colors[1],
    'Split Miner': colors[2],
    'Heuristics Miner': colors[3],
    'Inductive Miner': colors[4],
    'Alpha Miner': colors[5]
  }[algorithm]

def get_zorder(algorithm):
  return {
    'Our approach': 6,
    'ILP Miner': 4,
    'Split Miner': 5,
    'Heuristics Miner': 3,
    'Inductive Miner': 2,
    'Alpha Miner': 1
  }[algorithm]

take_all = True

# fig, ax = plt.subplots(figsize=(16, 8.62))
plot_type = 'scatter'
if plot_type == 'scatter':
  rows, cols = len(X), 6
  fig, ax = plt.subplots(rows, cols, figsize=(15, 5))

  min_x, min_y = 1, 1
  max_x, max_y = 0, 0
  for dataset_index, (dataset_name, algorithm_conformance) in enumerate(datasets.items()):
    # if dataset_name == 'Average':
    #   print(algorithm_conformance)
    #   print(a)
    row, col = int(dataset_index / cols), dataset_index % cols
    ax[row, col].set_title(labels.get(dataset_name, dataset_name))
    ax[row, col].set_xlabel(X_labels[X[1]], fontsize=13)
    ax[row, col].set_ylabel(X_labels[X[0]], fontsize=13)
    for algorithm_index, (algorithm, conformance) in enumerate(algorithm_conformance.items()):
      if isinstance(conformance[0], list):
        print(conformance)
        conformance = np.array(conformance)
        ax[row, col].scatter([np.mean(conformance[:,1])], [np.mean(conformance[:,0])], marker=markers[dataset_index], linewidth=1,
                             edgecolor='k',
                             color=get_color(get_algo_label(algorithm)), alpha=.7, s=200,
                             zorder=get_zorder(get_algo_label(algorithm)))
        continue
      if conformance[1] == 0 or conformance[0] == 0:
        continue
      ax[row, col].scatter([conformance[1]], [conformance[0]], marker=markers[dataset_index], linewidth=1, edgecolor='k',
                color=get_color(get_algo_label(algorithm)), alpha=.7, s=200, zorder=get_zorder(get_algo_label(algorithm)))
      min_x, min_y = min(min_x, conformance[1]), min(min_y, conformance[0])
      max_x, max_y = max(max_x, conformance[1]), max(max_y, conformance[0])
      if dataset_index == 0:
        ax[0, 4].scatter([10], [10], marker='o', linewidth=1.5, edgecolor='k',
                   color=get_color(get_algo_label(algorithm)), alpha=.9, s=180, label=f'{get_algo_label(algorithm)}')
        ax[0, 4].set_ylim(min_y - 0.05, 1.25)
        ax[0, 4].legend()
        ax[row, col].set_xlim(min_x - 0.05, 1.05)

  for i in range(12):
    row, col = int(i / cols), i % cols
    ax[row, col].set_ylim(min_y - 0.05, 1.05)
    ax[row, col].set_xlim(min_x - 0.05, 1.05)

elif plot_type == 'bar':
  rows, cols = len(X), 1
  fig, ax = plt.subplots(rows, cols, figsize=(15, len(X)*1.2 + 0.7))

  labelsies = [labels.get(dataset_name, dataset_name) for dataset_name in datasets.keys()]
  # conformances1, conformances2, conformances3, conformances4, conformances5 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
  conformances = [defaultdict(list) for i in range(len(X))]

  for dataset_index, (dataset_name, algorithm_conformance) in enumerate(datasets.items()):
    print(dataset_name)
    print(algorithm_conformance.keys())
    for algorithm_index, algorithm in enumerate(['gcn', 'split_reduced', 'heuristics_reduced', 'ilp_reduced', 'inductive_reduced']):
      print(algorithm_conformance)
      conformance = algorithm_conformance.get(algorithm, [0.0]*len(X))
      if algorithm == 'ilp_reduced':
        continue

      print('conformance', conformance)

      # for i, confie in enumerate(conformance):
      #   if not isinstance(confie, list):
      #     conformances[i][algorithm].append(confie)
      #   else:
      #     conformance = np.array(conformance)
      #     print(conformance[:, 0].nonzero()[0])
      #     conformances[i][algorithm].append(0.0) #np.mean([v for v in conformance[:, i] if v > 0]))

      if not isinstance(conformance[0], list):
        for i in range(len(X)):
          conformances[i][algorithm].append(conformance[i])
          # conformances[1][algorithm].append(conformance[1])
        # conformances3[algorithm].append(conformance[2])
      else:
        conformance = np.array(conformance)
        for i in range(len(X)):
          conformances[i][algorithm].append(np.mean([v for v in conformance[:,i] if v > 0]))
        # conformances[1][algorithm].append(np.mean([v for v in conformance[:,1] if v > 0]))

  width = 0.2  # the width of the bars
  offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

  x = np.arange(len(labelsies))
  for i, confies in enumerate(conformances):
    for index, (algorithm, scores) in enumerate(confies.items()):
      ax[i].bar(x + offsets[index], scores, width, label=get_algo_label(algorithm) if i == 0 else None, alpha=.7, edgecolor='k', color=get_color(get_algo_label(algorithm)))

  # for index, (algorithm, scores) in enumerate(conformances1.items()):
  #   print(len(scores))
  #   ax[0].bar(x + offsets[index], scores, width, label=f'{get_algo_label(algorithm)}', alpha=.7, edgecolor='k', color=get_color(get_algo_label(algorithm)))
  # for index, (algorithm, scores) in enumerate(conformances2.items()):
  #   ax[1].bar(x + offsets[index], scores, width, label=f'{get_algo_label(algorithm)}', alpha=.7, edgecolor='k', color=get_color(get_algo_label(algorithm)))


  for i in range(len(X)):
    ax[i].set_xticks(x)
    ax[i].set_ylabel(X_labels[X[i]], fontsize=13)
    if i == len(X) - 1:
      ax[i].set_xticklabels(labelsies, fontsize=13)
    else:
      ax[i].set_xticklabels([''] * len(labelsies), fontsize=13)

    ax[i].set_xlim(-0.5, 11.5)
    ax[i].set_ylim(0.24, 1.01)

  # ax[0].set_xticks(x)
  # ax[0].set_ylabel(X_labels[X[0]])
  # ax[1].set_ylabel(X_labels[X[1]])
  # ax[0].set_xticklabels([''] * len(labelsies))
  # ax[1].set_xticks(x)
  # # ax[0].set_xticklabels([''] * len(labelsies))
  # ax[1].set_xticklabels(labelsies)

  # ax[0].set_yticks([0.25, 0.5, 0.75, 1.0])
  # ax[1].set_yticks([0.5, 0.75, 1.0])

  # ax[0].set_yticks([0.5, 0.75, 1.0])
  # ax[1].set_yticks([0.25, 0.5, 0.75, 1.0])
  #
  # ax[0].set_ylim(0.24, 1.01)
  # ax[1].set_ylim(0.45, 1.01)
  # ax[1].set_ylim(0.24, 1.01)
  # ax[0].set_ylim(0.49, 1.01)
  # ax[1].set_ylim(0.09, 1.01)

  # ax[0].set_xlim(-0.5, 11.5)
  # ax[1].set_xlim(-0.5, 11.5)
  # ax[0].legend(ncol=4, bbox_to_anchor=(0.5,1.3), loc="upper center")
  # plt.xticks(rotation=15, ha='right')

# ax.set_ylim(min_y - 0.01, 1.01)
# ax.set_xlim(min_x - 0.01, 1.01)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=(6 if take_all else 5), fancybox=False, fontsize=13)

# fig.savefig(f'/home/dominique/TUe/thesis/presentation/figures/real_scatter_Fitness_Precision.pdf')
# fig.savefig(f'/home/dominique/TUe/thesis/report/eval_figures/real_scatter_{X[0]}_{X[1]}{"_all" if take_all else ""}.pdf')
# fig.savefig(f'/home/dominique/TUe/thesis/presentation/figures/results_scatter_fscore_generalization.pdf')

plt.tight_layout()
plt.subplots_adjust(hspace=0.1, top=0.92)

# fig.savefig(f'/home/dominique/TUe/thesis/paper/real_scatter_bar_{X[0]}_{X[1]}.pdf')
if args.export:
  fig.savefig('/mnt/c/Users/s140511/tue/thesis/paper/figures/real_fscore_simplicity.pdf')

plt.show()