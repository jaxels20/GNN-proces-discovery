import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import json
from functools import reduce
import operator
import os
from copy import deepcopy
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str)
parser.add_argument('-r', '--ratios', action='store_true')
parser.add_argument('-e', '--export', action='store_true')
parser.add_argument('-pt', '--print_table', action='store_true')
parser.add_argument('-f', '--filter_ones', action='store_true')
parser.add_argument('-c', '--category', type=str)
parser.add_argument('-bd', '--basedir', type=str)
args = parser.parse_args()


def get_conformance_data(filename):
  with open(filename, 'r') as training_file:
    reader = csv.DictReader(training_file)
    print(reader.fieldnames)
    conformances = defaultdict(dict)
    for row in reader:
      ff = row['name'].split('/')[1]
      dataset = row['name'].split('/')[0]

      try:
        row['entropia_fscore'] = (2 * float(row['entropia_recall']) * float(row['entropia_precision'])) / (
            float(row['entropia_recall']) + float(row['entropia_precision']))
      except ValueError:
        row['entropia_fscore'] = ''
      except ZeroDivisionError:
        row['entropia_fscore'] = ''

      if ff == 'gcn_sound':
        conformances[dataset]['prediction'] = row
      elif ff == 'groundtruth':
        conformances[dataset]['true'] = row
      elif ff == 'split':
        # if row['name'].split('/')[-1] in conformances:
        conformances[dataset]['split'] = row
      elif ff == 'inductive':
        # if row['name'].split('/')[-1] in conformances:
        conformances[dataset]['inductive'] = row
      elif ff == 'heuristics':
        # if row['name'].split('/')[-1] in conformances:
        conformances[dataset]['heuristics'] = row
      elif ff == 'ilp':
        # if row['name'].split('/')[-1] in conformances:
        conformances[dataset]['ilp'] = row
  return conformances


# basedir = '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_medium_ws2/logs/predictions'
# print(f'{basedir}/training_conformance.csv')
# # training_conformances = get_conformance_data(f'{basedir}/training_conformance.csv')
# test_conformances = get_conformance_data(f'{basedir}/all_conformance_final.csv')
# print(len(test_conformances.keys()))

def get_fscore(fitness, precision):
  try:
    return (2 * fitness * precision) / (fitness + precision)
  except:
    return np.nan

def cc_sum(cc):
  stats = deepcopy(cc)
  if 'FULL' in cc:
    stats['full'] = cc['FULL']
  categories = {
    'fitness': (('full', 'fitness', 'log_fitness'), np.nan),
    'precision': (('full', 'precision'), np.nan),
    'simplicity': (('full', 'simplicity'), np.nan),
    'generalization': (('full', 'generalization'), np.nan),
    'metricsAverageWeight': (('full', 'metricsAverageWeight'), np.nan),
    'fitness_alignments': (('full', 'fitness_alignments', 'averageFitness'), np.nan),
    'precision_alignments': (('full', 'precision_alignments'), np.nan),
    'entropia_recall': (('full', 'entropia_recall'), np.nan),
    'entropia_precision': (('full', 'entropia_precision'), np.nan),
    'sound': (('sound',), False),
    'easy_soundness': (('easy_soundness',), False),
  }
  cc_sum = {}
  for cat, (keys, defaultvalue) in categories.items():
    try:
      cc_sum[cat] = reduce(operator.getitem, keys, stats)
      if cat == 'fitness_alignments' and isinstance(cc_sum[cat], float) and cc_sum[cat] <= 0:
        cc_sum[cat] = float(cc_sum['full']['fitness_alignments']['percFitTraces']) / 100

      if isinstance(cc_sum[cat], float):
        cc_sum[cat] = max(min(cc_sum[cat], 1.0), 0.0)
    except KeyError as e:
      # print(e)
      cc_sum[cat] = defaultvalue

  cc_sum['fscore'] = get_fscore(cc_sum['fitness'], cc_sum['precision'])
  cc_sum['entropia_fscore'] = get_fscore(cc_sum['entropia_recall'], cc_sum['entropia_precision'])
  cc_sum['fscore_alignments'] = get_fscore(cc_sum['fitness_alignments'], cc_sum['precision_alignments'])

  return cc_sum

def get_conformance_data2(cc_dir, samples, methods):
  conformances = {}
  for sample in samples:
    print(sample)
    sample =f'{sample:04d}'
    conformances[sample] = {}
    for method in methods:
      verbose = 'gcn' in method

      if 'gcn' in method:
        cc_filename = f'{cc_dir}/{sample}_{method}_cca.json'
      else:
        cc_filename = f'{cc_dir}/{sample}_{method}_cce.json'
      if os.path.exists(cc_filename):
        with open(cc_filename, 'r') as f:
          cc = json.load(f)
          pprint(cc)
      else:
        cc = {}
      if 'FULL' in cc:
        cc['full'] = cc['FULL']

      if 'gcn' in method:
        method = 'prediction'
      elif method == 'groundtruth':
        method = 'true'
        print(cc)
        if cc['full']['fitness_alignments']['averageFitness'] == 0:
          cc['full']['fitness_alignments']['averageFitness'] = 1

      conformances[sample][method] = cc_sum(cc)
      if verbose:
        if conformances[sample][method]['fitness_alignments'] == 0:
          print(sample)
          pprint(conformances[sample][method])
          print(fdsa)

      # if method == 'true' and conformances[sample][method]['fitness_alignments'] == 0:
      #   # TODO FIX THIS!
      #   print(sample)
      #   break
  return conformances

dataset = args.dataset
cc_dir = {'simple': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions',
          'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws/predictions',
          'small': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees/predictions',}[dataset]
duplicates = {'simple': [2010, 2027, 2047, 2263, 2359, 2492],
              'complex': [1494, 1545, 1553, 1600, 1604, 1622, 1625, 1644, 1649, 1659, 1670, 1672, 1677, 1690, 1705, 1721, 1755, 1758, 1778, 1797, 1824, 1825, 1846, 1863, 1884, 1892, 1909, 1970, 1973],
              'small': []}[dataset]
samples = {'simple': range(1970, 2627), 'complex': range(1488, 1985), 'small': range(0, 500)}[dataset]
mf_model = {'simple': 'gcn_ep144b', 'complex': 'gcn_ep100b', 'small': 'gcn33'}[dataset]
methods = ['groundtruth', 'split', 'heuristics', 'inductive']
methods.insert(1, mf_model)
methods = [mf_model]

test_conformances = get_conformance_data2(cc_dir, [v for v in samples if v not in duplicates], methods)
pprint(test_conformances)


# for dataset, result in test_conformances.items():
#   print(dataset, result['prediction'])
# print(a)

def get_per_category(conformances):
  dicts = {
    'true': defaultdict(lambda: np.array([])),
    'prediction': defaultdict(lambda: np.array([])),
    'split': defaultdict(lambda: np.array([])),
    'inductive': defaultdict(lambda: np.array([])),
    'heuristics': defaultdict(lambda: np.array([])),
    # 'ilp': defaultdict(lambda: np.array([]))
  }

  for dataset, conformance in conformances.items():
    # if sum([method not in conformance for method in dicts.keys()]) != 0:
    #   print('fdsafdsafdsaffdsafdsafdsaffdsafdsafdsaffdsafdsafdsaf')
    #   print(fdsa)
    #   break
    for method, dictionary in dicts.items():
      if method != 'prediction':
        continue
      for key, value in conformance[method].items():
        if key != 'name':
          if value in ['n/a', 'nan', '0.0', '0', '', None]:
            dictionary[key] = np.append(dictionary[key], None)
          else:
            try:
              dictionary[key] = np.append(dictionary[key], float(value))
            except ValueError:
              v = False if value == 'False' else True
              dictionary[key] = np.append(dictionary[key], v)
  return dicts['true'], dicts['prediction'], dicts['split'], dicts['inductive'], dicts['heuristics'] #, dicts['ilp']

# training_trues, training_predictions, splits = get_per_category(training_conformances)
# print(training_trues['fscore'].shape)
# print(training_predictions['fscore'].shape)

test_trues, test_predictions, test_splits, test_inductives, test_heuristics = get_per_category(test_conformances)
test = {
  'Ground truth': test_trues, 'Our approach': test_predictions, 'Split Miner': test_splits,
  'Inductive Miner': test_inductives, 'Heuristics Miner': test_heuristics #, 'ILP Miner': test_ilps
}

# print(len(test['Our approach']['fscore']))
# print(A)

print('fscores')
for name in ['Our approach', 'Ground truth', 'Split Miner', 'Heuristics Miner', 'Inductive Miner']:
  print(f'{name:<17}: {test[name]["fscore"].shape}')

for key in ['soundness', 'easy_soundness']:
  print('\n', key)
  for name in ['Our approach']:  # , 'Ground truth', 'Split Miner', 'Heuristics Miner', 'Inductive Miner']:
    print([i for i, x in enumerate(test[name][key]) if x == 1.])
    print(f'{name:<17}: {sum(test[name][key])} {sum(test[name][key]) / 663 * 100:.2f}%')


if args.print_table:
  from rich.pretty import pprint
  pprint(test['Split Miner'].keys())

  data = {}
  for method, cc in test.items():
    data[method] = []
    for key in ['entropia_recall' ,'entropia_precision' ,'entropia_fscore' ,'fitness_alignments' ,'precision_alignments' ,'fscore_alignments' ,'simplicity']:
      print(key)
      non_nuls = [v for i, v in enumerate(cc[key])
                  if v is not None and not np.isnan(v) and test['Our approach']['easy_soundness'][i] == 1.]
      non_nuls = [v for v in non_nuls if v > 0]
      print(len(non_nuls), len(non_nuls))

      # if method == 'Ground truth':
      #   print(method)
      #   print(cc.keys())
      #   print(cc[key])
      #   print((np.mean(non_nuls), np.median(non_nuls)))
      #   print(fdsa)
      data[method].append((np.mean(non_nuls), np.median(non_nuls)))

  from temp_table import table_test
  table_test(data)
  print(fdsa)


def filter(data):
  data = data[data is not None]
  if args.filter_ones:
    return data[np.where(data != 1)]
  else:
    return data


def histogram(stats, bin_step=.04, bins=None, labels=None):
  if labels is None:
    labels = [''] * len(stats)

  original_lengths = [len(stat) for stat in stats.values()]

  if args.filter_ones:
    stats = {key: [value for value in stat if value != 1] for key, stat in stats.items()}
    print('filtering!')
    # stats = [filter(stat) for stat in stats]

  weights = [np.ones_like(stat) / length for stat, length in zip(stats.values(), original_lengths)]
  if bins is None:
    maxvalue = max([max(st) for st in stats.values()])
    minvalue = min([min(st) for st in stats.values()])
    maxvalue = 1.5
    bins = np.arange(minvalue - (minvalue % bin_step) - 0.5 * bin_step,
                     maxvalue - (maxvalue % bin_step) + 1.5 * bin_step, bin_step)

  ax.hist(stats.values(), alpha=0.7, bins=bins, align='mid', color=[get_color(key) for key in stats.keys()],
          label=labels, edgecolor='k')  # , weights=weights)


plt.style.use('ggplot')
# fig, ax = plt.subplots(figsize=(9, 3))

bins = np.linspace(0.6, 2, 21)
bins = None

print(bins)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_color(algorithm):
  return {
    'Our approach': colors[0],
    'ILP Miner': colors[1],
    'Split Miner': colors[2],
    'Heuristics Miner': colors[3],
    'Inductive Miner': colors[4],
    'Ground truth': colors[5],

  }[algorithm]


def get_zorder(algorithm):
  return {
    'Our approach': 6,
    'ILP Miner': 2,
    'Split Miner': 4,
    'Heuristics Miner': 3,
    'Inductive Miner': 5,
    'Ground truth': 3.5
  }[algorithm]


X_labels = {
  'fscore': 'F-score_',
  'fitness': 'Fitness_',
  'precision': 'Precision_',
  'simplicity': 'Simplicity',
  'generalization': 'Generalization',
  'metricsAverageWeight': 'metricsAverageWeight',
  'fscore_alignments': 'F-score (alignment)',
  'fitness_alignments': 'Fitness (alignment)',
  'precision_alignments': 'Precision (alignment)',
  'entropia_recall': 'Fitness (entropy)',
  'entropia_precision': 'Precision (entropy)',
  'entropia_fscore': 'F-Score (entropy)'
}

category = args.category
# if args.ratios:
#   histogram([training_predictions[category] / training_trues[category],
#              test_predictions[category] / test_trues[category]], labels=['Training data predictions/trues', 'Test data predictions/trues'], bins=bins)
# else:
#   histogram(
#     [filter(training_predictions[category]), filter(training_trues[category]), filter(test_predictions[category]), filter(test_trues[category])],
#     labels=['Training data predictions', 'Test data predictions', 'Training data trues', 'Test data trues'], bins=bins)

histogrammie = False
if histogrammie:
  category = 'fscore_alignments'
  category = 'entropia_fscore'
  names = ['Ground truth', 'Split Miner', 'Heuristics Miner', 'Inductive Miner']
  datas = {}
  skip_indices = []
  soundies = 0
  easy_soundies = 0
  for name in names:
    data = []
    indices = []
    for index, (soundness_ours, easysoundness_ours, v_ours, soundness_other, v_other) in enumerate(
      zip(test['Our approach']['soundness'], test['Our approach']['easy_soundness'], test['Our approach'][category],
          test[name]['soundness'], test[name][category])):
      v_ours = None if v_ours is None or np.isnan(v_ours) else v_ours
      v_other = None if v_other is None or np.isnan(v_other) else v_other
      if name == 'Ground truth' and soundness_ours == 1.0:
        soundies += 1
      if name == 'Ground truth' and easysoundness_ours == 1.0:
        easy_soundies += 1

      indices.append(index)
      if v_ours is not None and v_other is not None:
        data.append(v_ours / v_other)
    datas[name] = data
    print(name, 'l', len(data))
  print(soundies, easy_soundies)
  [print(max(r)) for r in datas]

  # datas = [filter(test_trues[category]), filter(test_predictions[category]), filter(test_splits[category]),
  #          filter(test_heuristics[category]), filter(test_inductives[category]), filter(test_ilps[category])]
  for name, data in zip(names, datas.values()):
    print(
      f'{name:<16}: mean {np.mean(data):.3f}, min {np.min(data):.3f}, max {np.max(data):.3f}, median {np.median(data):.3f}')

  bin_step = 0.04
  histogram(datas, labels=names, bins=bins, bin_step=bin_step)

  ax.set_xticks(np.arange(0 - bin_step, 2, bin_step))
  ax.set_xlim(0.7, 1.3)
  ax.set_ylim(0, 175)
  ax.set_xlabel('F-score ratio (Our approach / <method>)')
  plt.tight_layout()
  plt.subplots_adjust(top=0.977, bottom=0.160)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=(6), fancybox=False, fontsize=11)

  # fig.savefig(f'/home/dominique/TUe/thesis/paper/conformance_training_{category}_ratios.pdf')

else:
  plt.style.use('ggplot')
  fig, ax = plt.subplots(2, 2, figsize=(9, 8))
  # fig.suptitle(args.dataset)

  names = ['Our approach', 'Ground truth', 'Split Miner', 'Heuristics Miner', 'Inductive Miner']  # , 'ILP Miner']
  axes = [['entropia_precision', 'entropia_recall'], ['precision_alignments', 'fitness_alignments'],
          ['simplicity', 'entropia_fscore'], ['simplicity', 'fscore_alignments']]
  # x = 'fscore_alignments'
  # y = 'simplicity'
  # x = 'entropia_fscore'
  # y = 'simplicity'
  # x = 'entropia_recall'
  # y = 'entropia_precision'
  # x = 'fitness_alignments'
  # y = 'precision_alignments'
  # fscores, simplicities = [], []

  skip_indices = []
  xs = [[], [], [], []]
  ys = [[], [], [], []]
  testels = []
  for i, (x, y) in enumerate(axes):
    for name in names:
      print(name)
      values_x, values_y = [], []
      for index, (soundness, fscore, fscore_alignments, v1, v2) in enumerate(
        zip(test[name]['easy_soundness'], test[name]['fscore'], test[name]['fscore_alignments'], test[name][x], test[name][y])):
        if index in skip_indices:
          continue
        if name == 'Our approach' and (soundness == 0.0 or fscore_alignments is None or v1 is None):
          print('skip_incices', skip_indices)
          skip_indices.append(index)
          continue
        if v1 is not None and v2 is not None:
          values_x.append(v1)
          values_y.append(v2)
        testels.append([index, fscore_alignments, soundness])
      print(len(values_y))
      # break
      xs[i].append(np.array(values_x))
      ys[i].append(np.array(values_y))

  # print(sorted(testels, key=lambda x: x[1]))
  # print('worst', [x[0] for x in sorted(testels, key=lambda x: x[1])[:10]])
  # print('best', [x[0] for x in sorted(testels, key=lambda x: x[1], reverse=True)[:10]])

  # fscores = [filter(test_trues['fscore']), filter(test_predictions['fscore']), filter(test_splits['fscore']),
  #            filter(test_heuristics['fscore']), filter(test_inductives['fscore']), filter(test_ilps['fscore'])]
  # simplicities = [filter(test_trues['simplicity']), filter(test_predictions['simplicity']), filter(test_splits['simplicity']),
  #                 filter(test_heuristics['simplicity']), filter(test_inductives['simplicity']), filter(test_ilps['simplicity'])]

  for index, (xx, yy, labels) in enumerate(zip(xs, ys, axes)):
    x_label, y_label = labels
    print('x_label', x_label, 'y_label', y_label)
    axie = ax[int(index / 2)][index % 2]
    for name, xxx, yyy in zip(names, xx, yy):
      # if name != 'Ground truth':
      #   continue
      print(name)
      print(xxx.shape)
      print(yyy.shape)
      axie.plot(xxx, yyy, marker='o', linestyle='', color=get_color(name), alpha=.6, ms=6,
              zorder=get_zorder(name))

      nonnans = ~np.isnan(yyy) & ~np.isnan(xxx)
      axie.scatter([np.mean(xxx[nonnans])], [np.mean(yyy[nonnans])], marker='o', linewidth=1.2, edgecolor='k', color=get_color(name), alpha=.9, s=180, zorder=get_zorder(name) + 6, label=name if index == 0 else None)

      axie.scatter([np.median(xxx[nonnans])], [np.median(yyy[nonnans])], marker='v', linewidth=1.2, edgecolor='k',
                 color=get_color(name), alpha=.9, s=180, zorder=get_zorder(name) + 6)

      if name == 'Our approach' and index == 0:
        ax[0][0].scatter([10], [10], marker='v', linewidth=1.2, edgecolor='k', color='w', alpha=.9, s=180, label='median')
      elif name == 'Ground truth' and index == 0:
        ax[0][0].scatter([10], [10], marker='o', linewidth=1.2, edgecolor='k', color='w', alpha=.9, s=180, label='mean')


    axie.set_xlabel(X_labels[x_label])
    axie.set_ylabel(X_labels[y_label])
    # axie.set_ylim(0.75, 1.01)
    # axie.set_xlim(0.75, 1.01)
    axie.set_ylim(0.75, 1.01)
    axie.set_xlim(0.75, 1.01)


  # ax.plot([10], [10], color=get_color(name), label=f'{name}')

  # THIS ONE
  # ax.set_ylim(0.75, 1.035)
  # ax.set_xlim(0.75, 1.01)
  #### ax.set_ylim(0.6, 1.035)
  #### ax.set_xlim(0.6, 1.01)

  # ax.set_ylim(0.6, 1.01)
  # ax.set_xlim(0.1, 1.01)

  # ax.set_ylabel('FScore (Fitness+Precision)')
  # ax.set_xlabel('Generalization')
  #### ax.set_ylabel(X_labels[x])
  #### ax.set_xlabel(X_labels[y])

  # plt.legend(loc='lower left')
  fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=(5), fancybox=False, fontsize=11)
  plt.tight_layout()
  plt.subplots_adjust(top=0.915)


  # handles, labels = plt.gca().get_legend_handles_labels()
  # order = [0,5,2,6,4,1,3]
  # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.46, 1.03), ncol=(5), fancybox=False, fontsize=11)

# fig.savefig(f'/mnt/c/Users/s140511/tue/thesis/paper/figures/synthetic_scatter_{x}_{y}.pdf')
# fig.savefig(f'/mnt/c/Users/s140511/tue/thesis/paper/figures/synthetic_scatters.pdf')

if args.export:
  filename = f'/mnt/c/Users/s140511/tue/thesis/paper/figures/synthetic_scatters_{args.dataset}.pdf'
  print(f'Exporting {filename}')
  fig.savefig(filename)

plt.show()
