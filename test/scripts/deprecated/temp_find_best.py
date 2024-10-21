import os
import json
import tqdm
from copy import deepcopy
import numpy as np
from functools import reduce
import operator
from pprint import pprint
import shutil

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
      if isinstance(cc_sum[cat], float):
        cc_sum[cat] = max(min(cc_sum[cat], 1.0), 0.0)
    except KeyError as e:
      # print(e)
      cc_sum[cat] = defaultvalue

  cc_sum['fscore'] = get_fscore(cc_sum['fitness'], cc_sum['precision'])
  cc_sum['entropia_fscore'] = get_fscore(cc_sum['entropia_recall'], cc_sum['entropia_precision'])
  cc_sum['fscore_alignments'] = get_fscore(cc_sum['fitness_alignments'], cc_sum['precision_alignments'])

  return cc_sum

def eval_data(samples, methods, directory):
  for sample in tqdm.tqdm(samples):
    print('='*120)
    print(sample)
    sample = sample[0]
    # file_names = os.listdir(f'{directory}/{sample}/results')
    # # print(methods)
    # # print(file_names)
    # # print(fdsa)
    # for fn in file_names:
    #   print(fn)
    #   for method in methods:
    #     if method in fn:
    #       os.remove(f'{directory}/{sample}/results/{fn}')
    #       break
    #       print(f'removing {fn}')
    # continue
    # print(fdsa)



    log_filename = f"{directory}/{sample}/data.xes"
    best_fscore = -1
    best_i = -1
    ccs = {}
    for i, method in enumerate(methods):
      pnml_filename = f'{directory}/{sample}/predictions/data_{method}.pnml'
      png_filename = f'{directory}/{sample}/predictions/pngs/data_{method}.png'
      if not os.path.exists(pnml_filename):
        # print(f'{pnml_filename} file not exists.')
        continue

      cc_filename = f'{directory}/{sample}/predictions/data_{method}_cca.json'
      if not os.path.exists(cc_filename):
        # print(f'{cc_filename} file not exists.')
        continue

      # print(cc_filename)
      with open(cc_filename, 'r') as f:
        cc = cc_sum(json.load(f))
        ccs[i] = cc
        ccs[i]['pnml_filename'] = pnml_filename
        ccs[i]['cc_filename'] = cc_filename
        ccs[i]['png_filename'] = png_filename

    if len(ccs.keys()) == 0:
      print('NO SOLUTIONS FOUND.')
      continue

    keys = ['entropia_fscore', 'fscore_alignments', 'fscore']
    for key in keys:
      # print(key)
      values = {i: cc for i, cc in ccs.items() if key in cc.keys() and not np.isnan(float(cc[key]))}
      # pprint(values)
      if len(values) > 0:
        print(f'KEY={key}')
        best_i, best_cc = max(values.items(), key=lambda d: d[1][key])
        print(best_i)
        pprint(best_cc)
        print()
        shutil.copy(best_cc['pnml_filename'], best_cc['pnml_filename'].replace('/predictions/', '/results/'))
        shutil.copy(best_cc['cc_filename'], best_cc['cc_filename'].replace('/predictions/', '/results/'))
        shutil.copy(best_cc['png_filename'], best_cc['png_filename'].replace('/predictions/pngs/', '/results/'))
        break

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-ds', '--dataset', help='dataset', type=str)
  args = parser.parse_args()

  datasets = [
    ('BPI_2012_A', 17),
    ('BPI_2012_O', 30),
    ('BPI_2017_A', 30),
    ('BPI_2017_O', 30),
    ('BPI_2020_Domestic_declarations', 8),
    ('BPI_2020_International_declarations', 3),
    ('BPI_2020_Permit_log', 75),
    ('BPI_2020_Prepaid_travel_cost', 30),
    ('BPI_2020_Request_for_payment', 8),
    ('road_traffic_fine', 30),
    ('sepsis', 8)
  ]

  models = [
    'simpleep144',
    'complexep100'
  ]

  method = args.dataset
  mf_model = {'simple': 'gcn_simpleep144', 'complex': 'gcn_complexep100'}[method]
  samples = datasets
  methods = [f'{mf_model}_{i}' for i in range(20)]

  eval_data(samples, methods, '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data')