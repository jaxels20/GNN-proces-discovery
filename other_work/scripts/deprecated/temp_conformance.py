from project.data_handling.petrinet import PetrinetHandler
from project.data_handling.log import LogHandler
from project.evaluation.petrinets import PetrinetEvaluation
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.analysis import woflan
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter
import numpy as np
import os
import json
from colorama import Fore, Style
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', help='dataset', type=str)

args = parser.parse_args()


method = args.dataset
directory = {
  'simple': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2',
  'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws',
  'small': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees',
}[method]
mf_model = {'simple': 'gcn_ep144b', 'complex': 'gcn_ep100b', 'small': 'gcn33'}[method]
samples = {'simple': range(1970,2627), 'complex': range(1488,1985), 'small': range(0, 734)}[method]


def get_log(log_filename, topXTraces):
  log = xes_importer.apply(log_filename)

  variants_count = case_statistics.get_variant_statistics(log)
  variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
  variants = [variant['variant'] for variant in variants_count][:topXTraces]
  # [print(variant) for variant in variants]
  log_top_x = variants_filter.apply(log, variants)

  return log, log_top_x

def conformance(petrinet_handler, log, log_top_x, topXTraces, entropia, export=''):
  conformance_filename = f'{petrinet_handler.filename[:-5]}_cca.json'

  petrinet_evaluation = PetrinetEvaluation(petrinet_handler.mPetrinet, petrinet_handler.mInitialMarking, petrinet_handler.mFinalMarking)

  soundness = petrinet_handler.get_pm4py_soundness()
  if soundness[0]:
    easy_soundness = True
  else:
    easy_soundness = petrinet_handler.get_easy_soundness(timeout=15)

  print(f'Sound: {Fore.GREEN if soundness[0] else Fore.RED}{soundness[0]}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
  result = petrinet_evaluation.conformance(log, alignment_based=easy_soundness)
  result_top_x = petrinet_evaluation.conformance(log_top_x, alignment_based=easy_soundness)

  for key, value in entropia.items():
    result[key] = value

  # if best_conformance is None or float(best_conformance[0]['fscore']) < float(result['fscore']):
  #   best_conformance = result, i

  if export != '':
    print('EXPORTING')
    with open(conformance_filename, 'w') as cc_file:
      conformances = {'full': result, f'top {topXTraces}': result_top_x,
                      'sound': soundness[0], 'easy_soundness': easy_soundness, 'filename': conformance_filename}
      json.dump(conformances, cc_file, sort_keys=True, indent=2)
'''
  # if export != '':
  #   if len(results) > 1:
  #     conformance_filename = f'{dddd}/predictions/{petrinet_name}_gcn_{export}_{i}_cc.json'
  #     with open(conformance_filename, 'w') as cc_file:
  #       conformances = {'full': result, f'top {topXTraces}': result_top_x,
  #                       'sound': soundness[0],
  #                       's_components': [[str(el) for el in comp] for comp in soundness[1].get('s_components', [])],
  #                       'uncovered_places_s_component': [str(el) for el in
  #                                                        soundness[1].get('uncovered_places_s_component', [])],
  #                       'easy_soundness': easy_soundness, 'filename': conformance_filename}
  #       json.dump(conformances, cc_file, sort_keys=True, indent=2)
  #   else:
  #
  '''


for sample in tqdm.tqdm(samples):
# for sample in [1894]:
  print('#' * 120)
  print('#' * 120)
  print('#' * 120)
  print(sample)
  print('#' * 120)
  print('#' * 120)
  print('#' * 120)
  petrinet_filename = f'{directory}/predictions/{sample:04d}_{mf_model}.pnml'
  if not os.path.exists(petrinet_filename):
    print(f'Sample {sample} does not exist.')
    continue

  pnet = PetrinetHandler()
  pnet.importFromFile(petrinet_filename)
  pnet.filename = petrinet_filename

  pnet.removeStartAndEndTransitions()

  log_filename = f'{directory}/logs/{sample:04d}.xes'
  cc_filename = f'{directory}/predictions/{sample:04d}_{mf_model}_cce.json'
  if not os.path.exists(cc_filename):
    print('cc_filename does not exists')
    soundness = pnet.get_pm4py_soundness()
    print(soundness[0])
    entropia = {'entropia_precision': np.nan,
                'entropia_recall': np.nan}
    # continue
    top_x_traces = 30
  else:
    with open(cc_filename, 'r') as f:
      cc_data = json.load(f)
    entropia = {'entropia_precision': cc_data['full']['entropia_precision'], 'entropia_recall': cc_data['full']['entropia_recall']}
    top_x_traces = int([k for k in cc_data.keys() if 'top ' in k][0].split(' ')[1])
  print(top_x_traces)

  log, log_top_x = get_log(log_filename, top_x_traces)
  conformance(pnet, log, log_top_x, top_x_traces, entropia=entropia, export='yes')
  # break
