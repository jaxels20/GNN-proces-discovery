import json
import os
from pprint import pprint
from colorama import Style, Fore

from project.evaluation.petrinets import PetrinetEvaluation
from project.data_handling.petrinet import PetrinetHandler
from pm4py.objects.log.importer.xes import importer as xes_importer


find = True
conformance = not find

datasets = [
  'road_traffic_fine',
  'sepsis',
  'BPI_2012_A',
  'BPI_2012_O',
  'BPI_2017_A',
  'BPI_2017_O',
  'BPI_2020_Prepaid_travel_cost',
  'BPI_2020_Request_for_payment',
  'BPI_2020_Domestic_declarations',
  'BPI_2020_Permit_log',
  'BPI_2020_International_declarations',
]

methods = ['gcn', 'split', 'heuristics', 'inductive', 'ilp']
method_colors = [Fore.LIGHTRED_EX, Fore.LIGHTMAGENTA_EX, Fore.WHITE, Fore.YELLOW, Fore.LIGHTBLUE_EX]
def print_result(result):
  for method in methods:
    if method in result['filename'].split('/')[-1]:
      print(f'{method_colors[methods.index(method)]}{method}{Style.RESET_ALL}')

  print(f'{Fore.GREEN if result["sound"] else Fore.RED}{"not " if not result["sound"] else ""}sound - {Fore.GREEN if result["easy_soundness"] else Fore.RED}{"not " if not result["easy_soundness"] else ""}easy_sound{Style.RESET_ALL}')
  print(f'{"fscore":<9} | {"fitness":>9} | {"precision"} | {"generalization"} | {"simplicity"}')
  conf = result['full']
  fscore_string = f"{conf['fscore_alignments']:.5f}" if 'fscore_alignments' in conf else f"({conf['fscore']})"
  fitness_string = f"{conf['fitness_alignments']:.5f}" if 'fitness_alignments' in conf else f"({conf['fitness']})"
  precision_string = f"{conf['precision_alignments']:.5f}" if 'precision_alignments' in conf else f"({conf['precision']})"
  print(f'{fscore_string:>9} | {fitness_string:>9} | {precision_string:>9} | {conf["generalization"]:>14} | {conf["simplicity"]:<9}')

if find:
  for dataset in datasets:
    best_per_method = []
    print(f'{Fore.BLUE}{dataset}{Style.RESET_ALL}')
    for method in methods:
      print(f'{Fore.GREEN}{method}{Style.RESET_ALL}')
      base_dir = f'/home/dominique/TUe/thesis/git_data/evaluation_data/{dataset}/predictions/'
      files = [file for file in os.listdir(base_dir) if file[-4:] == '.txt' if method in file]

      results = []
      for file in files:
        with open(f'{base_dir}{file}', 'r') as f:
          try:
            result = {key: value for key, value in json.load(f).items() if key in ['sound', 'filename', 'full', 'easy_soundness']}
            result['full'] = {key.rstrip(): float(value) for key, value in result['full'].items()}
            if 'fitness_alignments' in result['full'] and 'precision_alignments' in result['full']:
              fscore = (2 * result['full']['fitness_alignments'] * result['full']['precision_alignments']) / (result['full']['fitness_alignments'] + result['full']['precision_alignments'])
              result['full']['fscore_alignments'] = fscore
            results.append(result)
          except json.decoder.JSONDecodeError:
            pass

      sorted_results = sorted(results, key=lambda x: int(x['sound']) + float(x['full'].get('fscore_alignments', '0')), reverse=True)
      # [pprint(r) for r in sorted_results]
      best_per_method.append(sorted_results[0])

      # sorted_results2 = sorted(results, key=lambda x: float(x['full'].get('fscore_alignments', '0')), reverse=True)
      # if sorted_results[0]['filename'] != sorted_results2[0]['filename']:
      #   pprint(sorted_results2[0])
    sorted_results = sorted(best_per_method, key=lambda x: float(x['full'].get('fscore_alignments', '0')), reverse=True)
    # [pprint(result) for result in sorted_results]
    [print_result(result) for result in sorted_results]

def print_conformance(result):
  scores = {
    f'{"fitness":<32}': f'{result["fitness"]["log_fitness"]:.5f}',
    f'{"precision":<32}': f'{result["precision"]:.5f}',
    f'{"generalization":<32}': f'{result["generalization"]:.5f}',
    f'{"fscore":<32}': f'{result["fscore"]:.5f}',
    f'{"metricsAverageWeight":<32}': f'{result["metricsAverageWeight"]:.5f}',
    f'{"simplicity":<32}': f'{result["simplicity"]:.5f}',
  }
  if 'fitness_alignments' in result:
    scores[f'{"fitness_alignments":<32}'] = f'{result["fitness_alignments"]["averageFitness"]:.5f}'
    scores[f'{"fitness_alignments_percFitTraces":<32}'] = f'{result["fitness_alignments"]["percFitTraces"]:.5f}'
  if 'precision_alignments' in result:
    scores[f'{"precision_alignments":<32}'] = f'{result["precision_alignments"]:.5f}'
  if 'fitness_alignments' in result and 'precision_alignments' in result and result["precision_alignments"] > 0 and result["fitness_alignments"]["averageFitness"] > 0:
    fscore = (2 * result["fitness_alignments"]["averageFitness"] * result["precision_alignments"]) / (result["fitness_alignments"]["averageFitness"] + result["precision_alignments"])
    scores[f'{"fscore_alignments":<32}'] = f'{fscore:.5f}'
  return scores

if conformance:
  datasets = ['road_traffic_fine']
  for dataset in datasets:
    base_dir = f'/home/dominique/TUe/thesis/git_data/evaluation_data/{dataset}/'
    log_filename = f'{base_dir}data.xes'
    # log = xes_importer.apply(log_filename)

    for method in ['gcn_beam_new_0']: #, 'heuristics', 'split', 'ilp']:
      # if method == 'split':
      pnml_filename = f'{base_dir}predictions/data_{method}.pnml'
      # else:
      #   pnml_filename = f'{base_dir}predictions/data_{method}_reduced.pnml'
      print(pnml_filename)

      petrinet_handler = PetrinetHandler()
      petrinet_handler.importFromFile(pnml_filename)

      petrinet_handler.visualize()
      print(a)

      if 'split' in pnml_filename:
        if len(petrinet_handler.mInitialMarking.keys()) == 0:
          for place in petrinet_handler.mPetrinet.places:
            if len(place.in_arcs) == 0:
              petrinet_handler.mInitialMarking[place] = 1
        if len(petrinet_handler.mFinalMarking.keys()) == 0:
          for place in petrinet_handler.mPetrinet.places:
            if len(place.out_arcs) == 0:
              petrinet_handler.mFinalMarking[place] = 1
      petrinet_handler.create_unique_start_place()

      petrinet_evaluation = PetrinetEvaluation(petrinet_handler.mPetrinet, petrinet_handler.mInitialMarking, petrinet_handler.mFinalMarking)
      soundness = petrinet_handler.get_pm4py_soundness()
      easy_soundness = petrinet_handler.get_easy_soundness(timeout=15)
      print(f'Sound: {Fore.GREEN if soundness[0] else Fore.RED}{soundness[0]}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
      result = petrinet_evaluation.conformance(log, alignment_based=easy_soundness, alignment_timeout=6000)

      conformance_filename = f'{pnml_filename[:-5]}_cc.txt'
      with open(conformance_filename, 'w') as cc_file:
        conformances = {'full': print_conformance(result),
                        'sound': soundness[0],
                        's_components': [[str(el) for el in comp] for comp in soundness[1].get('s_components', [])],
                        'uncovered_places_s_component': [str(el) for el in soundness[1].get('uncovered_places_s_component', [])],
                        'easy_soundness': easy_soundness, 'filename': conformance_filename}
        pprint(conformances)
        # json.dump(conformances, cc_file, sort_keys=True, indent=2, separators=(',', ': '))
