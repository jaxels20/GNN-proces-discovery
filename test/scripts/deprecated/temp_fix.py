from project.data_handling.petrinet import PetrinetHandler
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.analysis import woflan
import os
import json


dir = '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions - Copy'


# ccs_files = [f for f in os.listdir(dir) if 'gcn_ep100b_cc.json' in f]
ccs_files = [f for f in os.listdir(dir) if 'gcn_ep144b_cc.json' in f]
ccs = {}
for fn in ccs_files:
  i = int(fn[:4])
  with open(f'{dir}/{fn}', 'r') as f:
    ccs[i] = json.load(f)

count = 0
sound_count = 0
found_something = 0
for sample in range(1970,2627):
# for sample in range(1488, 1985):
  count += 1
  if sample not in ccs.keys():
    ccs[sample] = {'sound': False, 'found': False}
  else:
    ccs[sample]['found'] = True
    found_something += 1
    if not ccs[sample]['sound']:
      print(sample)
      break
  if ccs[sample]['sound']:
    sound_count += 1
print(count, found_something, sound_count)
print(fdsa)


# 497 308 152
# 497 302 145



method = 'complex'
directory = {
  'simple': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions',
  'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws/predictions'
}[method]
mf_model = {'simple': 'gcn_ep144b', 'complex': 'gcn_ep100b'}[method]
samples = {'simple': range(1970,2627), 'complex': range(1488,1985)}[method]


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


count = 0
sound_count = 0
found_something = 0
all_pngs = os.listdir(f'{directory}/pngs')
for sample in samples:
  count += 1
  petrinet_filename = f'{directory}/{sample:04d}_{mf_model}.pnml'
  if not os.path.exists(petrinet_filename):
    print(f'Sample {sample} does not exist.')
    continue
  found_something += 1

  png_filename = f'{directory}/pngs/{sample:04d}_{mf_model}.png'
  debug_png_filename = f'{directory}/pngs/{[png for png in all_pngs if f"{sample:04d}_{mf_model}_" in png][0]}'

  pnet = PetrinetHandler()
  pnet.importFromFile(petrinet_filename)
  fixed = fix_net(pnet)

  print(petrinet_filename)
  print(png_filename)
  print(debug_png_filename)
  if fixed:
    print(f'EXPORTING {sample}')
    pnet.export(petrinet_filename)
    pnet.visualize(fDebug=False, fExport=png_filename)
    pnet.visualize(fDebug=True, fExport=debug_png_filename)

  params = {'return_asap_when_not_sound': True, 'print_diagnostics': False, 'return_diagnostics': True}
  sound, diag = woflan.algorithm.apply(pnet.mPetrinet, pnet.mInitialMarking, pnet.mFinalMarking, params)
  if sound:
    sound_count += 1


print(count, found_something, sound_count)
print(fdsa)




pnfile = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data/BPI_2012_A/predictions/data_gcn_complexep100_0.pnml'

# pnfile = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions/2153_gcn_ep144b.pnml'

pnet = PetrinetHandler()
pnet.importFromFile(pnfile)

pnet.visualize()

first_tr = [t for t in pnet.mPetrinet.transitions if t.name == '>'][0]
first_place = list(pnet.mInitialMarking.keys())[0]
for index, out_arc in enumerate(first_tr.out_arcs):
  tr = petri_utils.add_transition(pnet.mPetrinet, f'>{index}', None)
  petri_utils.add_arc_from_to(first_place, tr, pnet.mPetrinet)
  petri_utils.add_arc_from_to(tr, out_arc.target, pnet.mPetrinet)
petri_utils.remove_transition(pnet.mPetrinet, first_tr)

last_tr  = [t for t in pnet.mPetrinet.transitions if t.name == '|'][0]
final_place = list(pnet.mFinalMarking.keys())[0]
for index, in_arc in enumerate(last_tr.in_arcs):
  tr = petri_utils.add_transition(pnet.mPetrinet, f'|{index}', None)
  petri_utils.add_arc_from_to(tr, final_place, pnet.mPetrinet)
  petri_utils.add_arc_from_to(in_arc.source, tr, pnet.mPetrinet)
petri_utils.remove_transition(pnet.mPetrinet, last_tr)

params = {'return_asap_when_not_sound': True, 'print_diagnostics': True, 'return_diagnostics': True}
print(woflan.algorithm.apply(pnet.mPetrinet, pnet.mInitialMarking, pnet.mFinalMarking, params))












if False:
  petri_utils.remove_place(pnet.mPetrinet, first_place)
  first_tr = list(first_place.out_arcs)[0].target
  first_tr.label = '>'
  del pnet.mInitialMarking[first_place]

  for a in first_tr.out_arcs:
    pnet.mInitialMarking[a.target] += 1

  final_place = list(pnet.mFinalMarking.keys())[0]
  print(final_place)
  final_tr = list(final_place.in_arcs)[0].source
  print(final_tr)

  final_places = [a.source for a in final_tr.in_arcs]
  final_tr.label = '|'
  # petri_utils.remove_transition(pnet.mPetrinet, final_tr)

  petri_utils.remove_place(pnet.mPetrinet, final_place)

  del pnet.mFinalMarking[final_place]
  print(pnet.mFinalMarking)

  for place in final_places:
    pnet.mFinalMarking[place] += 1

  pnet.merge_initial_final_marking()
  pnet.move_initial_final_markings()

pnet.visualize()
