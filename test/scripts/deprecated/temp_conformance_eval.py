from project.data_handling.petrinet import PetrinetHandler
from project.evaluation.petrinets import PetrinetEvaluation
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.analysis import woflan
import os
import tqdm
import json
from colorama import Fore, Style
from pprint import pprint
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter

datasets = [
  # ('BPI_2012_A', 17),
  # ('BPI_2012_O', 30),
  # ('BPI_2017_A', 30),
  # ('BPI_2017_O', 30),
  # ('BPI_2020_Domestic_declarations', 8),
  # ('BPI_2020_International_declarations', 3),
  # ('BPI_2020_Permit_log', 75),
  # ('BPI_2020_Prepaid_travel_cost', 30),
  # ('BPI_2020_Request_for_payment', 8),
  # ('road_traffic_fine', 30),
  ('sepsis', 8)
]


models = [
  'simpleep144',
  'complexep100'
]


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
    with open(conformance_filename, 'w') as cc_file:
      conformances = {'full': result, f'top {topXTraces}': result_top_x,
                      'sound': soundness[0], 'easy_soundness': easy_soundness, 'filename': conformance_filename}
      json.dump(conformances, cc_file, sort_keys=True, indent=2)




directory = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data'

stats = {
  'count': 0, 'found': 0, 'easy_sound': 0, 'sound': 0, 'new_easy_sound': 0, 'new_sound': 0
}
for dataset, top_x_traces in tqdm.tqdm(datasets):
  log_filename = f'{directory}/{dataset}/data.xes'
  log, log_top_x = get_log(log_filename, top_x_traces)

  for mf_model in tqdm.tqdm(models):
    for i in tqdm.tqdm(range(20)):
      stats['count'] += 1
      petrinet_filename = f'{directory}/{dataset}/predictions/data_gcn_{mf_model}_{i}.pnml'
      print(petrinet_filename)
      if not os.path.exists(petrinet_filename):
        print(f'Sample {petrinet_filename} does not exist.')
        continue
      stats['found'] += 1

      conformance_filename = f'{petrinet_filename[:-5]}_cce.json'
      with open(conformance_filename, 'r') as f:
        cc_data = json.load(f)
      entropia = {'entropia_precision': cc_data['full']['entropia_precision'],
                  'entropia_recall': cc_data['full']['entropia_recall']}

      pnet = PetrinetHandler()
      pnet.importFromFile(petrinet_filename)
      pnet.filename = petrinet_filename

      conformance(pnet, log, log_top_x, top_x_traces, entropia, export='yes')
      # print(fdsa)

      '''
      png_filename = f'{directory}/{dataset}/predictions/pngs/data_gcn_{mf_model}_{i}.png'
      debug_png_filename = f'{directory}/{dataset}/predictions/pngs/{[png for png in all_pngs if f"data_gcn_{mf_model}_{i}_" in png][0]}'
      print(debug_png_filename)

      pnet = PetrinetHandler()
      pnet.importFromFile(petrinet_filename)
      s, es = check_soundness(pnet)
      stats['sound'] += int(s)
      stats['easy_sound'] += int(es)
      # pnet.visualize()
      fixed = fix_net(pnet)
      s, es = check_soundness(pnet)
      stats['new_sound'] += int(s)
      stats['new_easy_sound'] += int(es)

      if True:
        print(f'EXPORTING data_{mf_model}_{i}')
        pnet.export(petrinet_filename)
        pnet.visualize(fDebug=False, fExport=png_filename)
        pnet.visualize(fDebug=True, fExport=debug_png_filename)
      '''

pprint(stats)

