from project.data_handling.petrinet import PetrinetHandler
from project.data_handling.log import LogHandler
from project.ml_models.preprocessing import GraphBuilder
from project.data_generation.generator import Generator
from project.data_generation.ptml_handling import read_ptml
from project.data_handling.petrinet import getPetrinetFromFile

from pm4py.objects.petri_net.utils import petri_utils

from copy import copy, deepcopy
import argparse
import json
import os
import time
import numpy as np
import pm4py
import tqdm

np.set_printoptions(linewidth=400)

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='Verbose output', action="store_true")
parser.add_argument('-e', '--export', help='Verbose output', action="store_true")
parser.add_argument('-v', '--visualize', help='Verbose output', action="store_true")
parser.add_argument('-n', '--number_of_traces', help='Number of traces', type=int)
parser.add_argument('-d', '--data_directory', help='dataDirectory', type=str)

args = parser.parse_args()

petrinet_dir     = f'{args.data_directory}/petrinets'
petrinet_png_dir = f'{args.data_directory}/petrinets_png'
ptml_dir         = f'{args.data_directory}/process_trees'
log_dir          = f'{args.data_directory}/logs'
log_compr_dir    = f'{args.data_directory}/logs_compressed'

def fix_start_end_silent_transition(petrinet_handler):
  silent_transitions = [transition for transition in petrinet_handler.mPetrinet.transitions if transition.label is None]
  for silent_transition in silent_transitions:
    if len(silent_transition.out_arcs) == 1 and len(silent_transition.in_arcs) == 1 and \
      list(silent_transition.in_arcs)[0].source in petrinet_handler.mInitialMarking and len(petrinet_handler.mInitialMarking) == 1:
      initial_place = list(petrinet_handler.mInitialMarking.keys())[0]
      token_count = petrinet_handler.mInitialMarking[initial_place]
      if len(initial_place.out_arcs) == 1:
        petrinet_handler.mInitialMarking[list(silent_transition.out_arcs)[0].target] = token_count
        del petrinet_handler.mInitialMarking[initial_place]
        petri_utils.remove_transition(petrinet_handler.mPetrinet, silent_transition)
        petri_utils.remove_place(petrinet_handler.mPetrinet, initial_place)
    if len(silent_transition.in_arcs) == 1 and len(silent_transition.out_arcs) == 1 and \
      list(silent_transition.out_arcs)[0].target in petrinet_handler.mFinalMarking and len(petrinet_handler.mFinalMarking) == 1:
      final_place = list(petrinet_handler.mFinalMarking.keys())[0]
      token_count = petrinet_handler.mFinalMarking[final_place]
      if len(final_place.in_arcs) == 1:
        petrinet_handler.mFinalMarking[list(silent_transition.in_arcs)[0].source] = token_count
        del petrinet_handler.mFinalMarking[final_place]
        petri_utils.remove_transition(petrinet_handler.mPetrinet, silent_transition)
        petri_utils.remove_place(petrinet_handler.mPetrinet, final_place)

def check_directly_follows(petrinet_handler, log_handler):
  # transitionLabels = ['>'] + [t.label for t in petrinet_handler.mPetrinet.transitions if t.label is not None] + [None, '|']
  transitionLabels = ['>'] + [t[0] for t in log_handler.mTransitions] + [None, '|']

  petrinet_handler.addStartAndEndTransitions()

  percentage = 80
  variants = log_handler.getMostFrequentVariants(percentage, minimum_variants=30, maximum_variants=30)
  traces = [list(variant['variant']) for variant in variants]
  counts = [variant['count'] for variant in variants]
  print(f'{len(log_handler.mVariants)} variants in original log, taking {len(traces)}.')

  if set([t.label for t in petrinet_handler.mPetrinet.transitions] + [None]) != set(transitionLabels):
    print('Not all transitions are found in the log.')
    return False

  pb = GraphBuilder(0, 10000, traces, counts, transitionLabels, fDepth=1, fPetrinetHandler=petrinet_handler, include_frequency=True)

  if np.sum(pb.mTarget) != len(petrinet_handler.mPetrinet.places):
    print(f'Not all places from the target petrinet have been constructed as candidate places. Got {np.sum(pb.mTarget)} out of {len(petrinet_handler.mPetrinet.places)}')
    return False

  petrinet_handler.removeStartAndEndTransitions()
  return True


sample_count = 0
i = 5
ptml_fns = sorted(os.listdir(ptml_dir), key=lambda s: int(s.split('.')[1])) #[i:i+1]
for ptml_fn in tqdm.tqdm(ptml_fns):
  print(ptml_fn)
  sample_name = f'{sample_count:04d}'

  pt = read_ptml(f'{ptml_dir}/{ptml_fn}')
  petrinet, m_i, m_f = pm4py.convert_to_petri_net(pt)
  petrinet_handler = PetrinetHandler()
  petrinet_handler.mPetrinet = petrinet
  petrinet_handler.mInitialMarking = m_i
  petrinet_handler.mFinalMarking = m_f
  petrinet_handler.label_silent_transitions(keepSafe=2)

  generator = Generator(sample_name, petrinet, m_i, m_f, {})
  log = generator.generateData(args.number_of_traces)
  log_handler = LogHandler(f'{log_dir}/{sample_name}', log)
  log_handler.getVariants()
  # print(log_handler.mVariants)

  # Remove silent transitions dangling from
  fix_start_end_silent_transition(petrinet_handler)
  petrinet_handler.remove_duplicate_silent_transitions()

  petrinet_handler_copy = petrinet_handler.copy()

  if args.visualize:
    petrinet_handler.visualize()

  nr_silent = len([t for t in petrinet_handler.mPetrinet.transitions if t.label is None])
  if args.export:
    petrinet_handler.export(f'{petrinet_dir}/{sample_name}.pnml')
    petrinet_handler.visualize(fExport=f'{petrinet_png_dir}/{sample_name}.png')

  # Check if number of non-silent transitions does not exceed max of 18?
  nonsilent_transitions = [t for t in petrinet_handler.mPetrinet.transitions if t.label is not None]
  usable = len(nonsilent_transitions) <= 18
  if not usable:
    print(f'Not usuable: too many transitions {len(nonsilent_transitions)} > 18')
  else:
    usable = check_directly_follows(petrinet_handler, log_handler)

  if not usable:
    if args.export:
      os.remove(f'{petrinet_dir}/{sample_name}.pnml')
      os.remove(f'{petrinet_png_dir}/{sample_name}.png')
    continue

  print('Good')

  ''' EXPORT '''
  if nr_silent > 0:
    print('-'*120)
    print(f'{nr_silent} silent transitions.')
    print('-' * 120)

  if args.export:
    print(f'EXPORTING: {sample_name}')
    log_handler.export_to_xes(log_handler.mLogFilename)
    log_handler.exportVariantsLog(f'{log_compr_dir}/{sample_name}.npz')

  sample_count += 1