from project.data_handling.petrinet import PetrinetHandler
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.analysis import woflan
import os
import tqdm
import json
from colorama import Fore, Style
from pprint import pprint

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

def fix_net(pnet):
  first_trs = [t for t in pnet.mPetrinet.transitions if t.name == '>']
  if len(first_trs) == 1:
    first_tr = first_trs[0]
    first_place = list(pnet.mInitialMarking.keys())[0]
    for index, out_arc in enumerate(first_tr.out_arcs):
      tr = petri_utils.add_transition(pnet.mPetrinet, f'>{index}', None)
      petri_utils.add_arc_from_to(first_place, tr, pnet.mPetrinet)
      petri_utils.add_arc_from_to(tr, out_arc.target, pnet.mPetrinet)
    petri_utils.remove_transition(pnet.mPetrinet, first_tr)
  else:
    print('ALREADY FIXED (s)')

  last_trs = [t for t in pnet.mPetrinet.transitions if t.name == '|']
  if len(last_trs) == 1:
    last_tr = last_trs[0]
    final_place = list(pnet.mFinalMarking.keys())[0]
    for index, in_arc in enumerate(last_tr.in_arcs):
      tr = petri_utils.add_transition(pnet.mPetrinet, f'|{index}', None)
      petri_utils.add_arc_from_to(tr, final_place, pnet.mPetrinet)
      petri_utils.add_arc_from_to(in_arc.source, tr, pnet.mPetrinet)
    petri_utils.remove_transition(pnet.mPetrinet, last_tr)
  else:
    print('ALREADY FIXED (l)')

  return len(first_trs) == 1 or len(last_trs) == 1

def check_soundness(pnet):
  soundness = pnet.get_pm4py_soundness()
  if soundness[0]:
    easy_soundness = True
  else:
    easy_soundness = pnet.get_easy_soundness(timeout=15)

  print(f'Sound: {Fore.GREEN if soundness[0] else Fore.RED}{soundness[0]}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
  return soundness[0], easy_soundness


directory = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data'

stats = {
  'count': 0, 'found': 0, 'easy_sound': 0, 'sound': 0, 'new_easy_sound': 0, 'new_sound': 0
}
for dataset, tx in tqdm.tqdm(datasets):
  for mf_model in tqdm.tqdm(models):
    all_pngs = [fn for fn in os.listdir(f'{directory}/{dataset}/predictions/pngs/')]
    for i in range(20):
      stats['count'] += 1
      petrinet_filename = f'{directory}/{dataset}/predictions/data_gcn_{mf_model}_{i}.pnml'
      if not os.path.exists(petrinet_filename):
        print(f'Sample {petrinet_filename} does not exist.')
        continue
      stats['found'] += 1

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
    # print(fdsa)

pprint(stats)




    # petrinet_filename = f'{directory}/{sample:04d}_{mf_model}.pnml'
    #
    # if not os.path.exists(petrinet_filename):
    #   print(f'Sample {sample} does not exist.')
    #   continue
    # found_something += 1
    #
    # png_filename = f'{directory}/pngs/{sample:04d}_{mf_model}.png'
    # debug_png_filename = f'{directory}/pngs/{[png for png in all_pngs if f"{sample:04d}_{mf_model}_" in png][0]}'
    #
    # pnet = PetrinetHandler()
    # pnet.importFromFile(petrinet_filename)
    # fixed = fix_net(pnet)
    #
    # print(petrinet_filename)
    # print(png_filename)
    # print(debug_png_filename)
    # if fixed:
    #   print(f'EXPORTING {sample}')
    #   pnet.export(petrinet_filename)
    #   pnet.visualize(fDebug=False, fExport=png_filename)
    #   pnet.visualize(fDebug=True, fExport=debug_png_filename)