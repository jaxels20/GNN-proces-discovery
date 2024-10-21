import numpy as np
import string
from colorama import Style, Fore
import argparse
from pprint import pprint
import tqdm

from project.evaluation.alpha_relations import AlphaRelations
from project.evaluation.places import PlaceEvaluation
from project.evaluation.petrinets import PetrinetEvaluation
from project.data_handling.petrinet import PetrinetHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', help='data directory', type=str)
parser.add_argument('-l', '--log_filename', help='log filename', type=str)
parser.add_argument('-vis', '--visualize', help='Visualize the petrinet', action='store_true')
parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
parser.add_argument('-s', '--simplify', help='Simplify the transition names', action='store_true')
parser.add_argument('-ase', '--add_start_and_end', help='Add start and end transition to true petrinet', action='store_true')


args = parser.parse_args()

number_of_mistakes = 0
dataset_name = args.log_filename

def add_statistics(statistics1, statistics2):
  for key, value in statistics1.items():
    for key2, value2 in value.items():
      statistics1[key][key2] += statistics2[key][key2]

combined_statistics = None

n_places = 0
n_places_true = 0
n_incorrect = 0
n_contains_fn = 0
n_contained_fn = 0
n_contains_tp = 0
n_contained_tp = 0
n_unknown = 0
for i in tqdm.tqdm(range(2000, 2663)):
# for i in tqdm.tqdm(range(2000, 2010)):
  dataset_name = f'{i:04d}'

  if args.data_directory is None:
    raise ValueError('data_directory argument should not be empty.')

  petrinet_handler = PetrinetHandler()
  try:
    petrinet_handler.importFromFile(f'{args.data_directory}/logs_compressed/predictions/{dataset_name}_gcn_sound.pnml')
  except OSError:
    continue
  petrinet_handler.merge_initial_final_marking()

  npz = np.load(f'{args.data_directory}/logs_compressed/{dataset_name}.npz', allow_pickle=True)

  variants = [list(variant) for count, variant in npz['variants']][:30]
  transition_names = [t.replace('\n', ' ') for t in npz['transitions'][:,0]]

  if args.simplify:
    simplified_mapping = {}
    for index, name in enumerate(sorted(transition_names)):
      simplified_mapping[name] = string.ascii_uppercase[index]

    transition_names = [simplified_mapping[name] for name in transition_names]

  alpha_relations = AlphaRelations(variants, transition_names)

  petrinet_handler_true = PetrinetHandler()
  petrinet_handler_true.importFromFile(f'{args.data_directory}/petrinets/{dataset_name}.pnml')
  if args.add_start_and_end:
    petrinet_handler_true.addStartAndEndTransitions()

  if args.simplify:
    petrinet_handler.simplify_transition_names()
    petrinet_handler_true.simplify_transition_names()
  if args.visualize:
    petrinet_handler.visualize(fDebug=True)
    petrinet_handler_true.visualize(fDebug=False)

  petrinet_evaluation = PetrinetEvaluation(petrinet_handler.mPetrinet, petrinet_handler.mInitialMarking, petrinet_handler.mFinalMarking)
  completely_correct = petrinet_evaluation.compare(petrinet_handler_true.mPetrinet, petrinet_handler_true.mInitialMarking, petrinet_handler_true.mFinalMarking)
  false_negatives = set([place for place in petrinet_handler_true.mPetrinet.places if not hasattr(place, 'marked')])
  for place in false_negatives:
    place.correct = False
    place.false_negative = True
  petrinet_evaluation.analyze_places(alpha_relations, places={*false_negatives, *petrinet_handler.mPetrinet.places})

  statistics = petrinet_evaluation.get_statistics(places={*false_negatives, *petrinet_handler.mPetrinet.places})
  if combined_statistics is None:
    combined_statistics = statistics
  else:
    add_statistics(combined_statistics, statistics)
  # print(combined_statistics)

  n_places += len(petrinet_handler.mPetrinet.places)
  n_places_true += len(petrinet_handler_true.mPetrinet.places)
  if not completely_correct[0]: # and len(false_negatives) > 0:
    if args.verbose: print(dataset_name)
    number_of_mistakes += 1
    if args.verbose:
      print(alpha_relations.get_matrix(names=True)[1])
      if args.verbose:petrinet_evaluation.print_analysis(verbosity=1)
    if len(false_negatives) > 0:
      if args.verbose:
        print(f'{Fore.BLUE}False negatives:{Style.RESET_ALL}')
        petrinet_evaluation.print_analysis(verbosity=1, places=false_negatives)

    # Check whether the false positives lie within the false negatives.
    if args.verbose: print(f'{Fore.BLUE}Mistakes evaluation:{Style.RESET_ALL}')
    for place in petrinet_handler.mPetrinet.places:
      if place.correct:
        continue
      n_false_negatives = len(false_negatives)
      n_incorrect += 1
      for false_negative in false_negatives:
        if place.evaluation.check_contained(false_negative):
          if args.verbose: print(f'{Fore.CYAN}{place} {place.input_transitions} {place.output_transitions} contained in false negative {false_negative} {false_negative.input_transitions} {false_negative.output_transitions}{Style.RESET_ALL}')
          n_contained_fn += 1
          break
        if false_negative.evaluation.check_contained(place):
          if args.verbose: print(f'{Fore.CYAN}{place} {place.input_transitions} {place.output_transitions} contains false negative {false_negative} {false_negative.input_transitions} {false_negative.output_transitions}{Style.RESET_ALL}')
          n_contains_fn += 1
          break
      else:
        for true_positive in petrinet_handler.mPetrinet.places:
          if place == true_positive or not true_positive.correct:
            continue
          if place.evaluation.check_contained(true_positive):
            if args.verbose: print(f'{Fore.CYAN}{place} {place.input_transitions} {place.output_transitions} contained in true positive {true_positive} {true_positive.input_transitions} {true_positive.output_transitions}{Style.RESET_ALL}')
            n_contained_tp += 1
            break
          if true_positive.evaluation.check_contained(place):
            if args.verbose: print(f'{Fore.CYAN}{place} {place.input_transitions} {place.output_transitions} contains true positive {true_positive} {true_positive.input_transitions} {true_positive.output_transitions}{Style.RESET_ALL}')
            n_contains_tp += 1
            break
        else:
          n_unknown += 1
          if args.verbose: print(f'{Fore.RED}{place} {place.input_transitions} {place.output_transitions} not contained.{Style.RESET_ALL}')
    if args.verbose: print()

print(f'Incorrect places: {n_incorrect}/{n_places} ({n_incorrect/n_places*100:.2f}%)')
print(f'Incorrect: {n_incorrect}')
print(f'n_places_true: {n_places_true}')
if n_incorrect > 0:
  print(f'of which {n_contained_fn} ({n_contained_fn / n_incorrect * 100:.2f}%) are contained in another correct false negative place and {n_contains_fn} ({n_contains_fn / n_incorrect * 100:.2f}%) contains another correct false negative place.')
  print(f'of which {n_contained_tp} ({n_contained_tp / n_incorrect * 100:.2f}%) are contained in another correct true positive place and {n_contains_tp} ({n_contains_tp / n_incorrect * 100:.2f}%) contains another correct true positive place.')
  print(f'{n_unknown} ({n_unknown / n_incorrect * 100:.2f}%) have unknown cause.')

for group, counts in combined_statistics.items():
  print(Fore.BLUE, group, Style.RESET_ALL)
  number_of_places = counts['#']
  for key, count in counts.items():
    print(f'{key:<20}: {count/number_of_places*100 if key != "#" else count:.1f}')
#
# print(f'number of mistakes: {number_of_mistakes}')

