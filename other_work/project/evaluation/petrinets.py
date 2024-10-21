from project.evaluation.places import PlaceEvaluation
from project.evaluation.entropia import Entropia

from pm4py.evaluation import evaluator
from pm4py import conformance
from pm4py.visualization.petrinet import visualizer as pn_visualizer

from pm4py.objects.petri import utils
from colorama import Fore, Style
import signal


class PetrinetEvaluation:
  def __init__(self, petrinet, initial_marking, final_marking):
    self.petrinet = petrinet
    self.initial_marking = initial_marking
    self.final_marking = final_marking

  def remove_unconnected_transitions(self):
    transitions_to_remove = [transition for transition in self.petrinet.transitions if len(transition.in_arcs) == 0 and len(transition.out_arcs) == 0]
    [utils.remove_transition(self.petrinet, transition) for transition in transitions_to_remove]

  def remove_start_end_transitions(self):
    transitions_to_remove = [transition for transition in self.petrinet.transitions if transition.name in ['>', '|']]
    [utils.remove_transition(self.petrinet, transition) for transition in transitions_to_remove]

  def compare_place_signatures(self, signature1, signature2):
    return signature1[0] == signature2[0] and signature1[1] == signature2[1]

  def get_place_signature(self, place):
    input_transitions = set([arc.source.label.replace('\n', ' ') for arc in place.in_arcs if arc.source.label is not None])  # if arc.source.label != '>'])
    output_transitions = set([arc.target.label.replace('\n', ' ') for arc in place.out_arcs if arc.target.label is not None])  # if arc.target.label != '|'])
    return input_transitions, output_transitions

  def compare(self, petrinet_true, initial_marking_true, final_marking_true):
    all_correct = True
    number_of_nodes = 0
    number_of_nodes_correct = 0
    for place in self.petrinet.places:
      number_of_nodes += 1
      place_signature = self.get_place_signature(place)
      place.correct = False
      for true_place in petrinet_true.places:
        true_place_signature = self.get_place_signature(true_place)
        if self.compare_place_signatures(place_signature, true_place_signature):
          place.correct = True
          true_place.marked = True
          number_of_nodes_correct += 1
          break
      else:
        all_correct = False

    for silent_transition in [t for t in self.petrinet.transitions if t.label is None]:
      place_in_signature = self.get_place_signature(list(silent_transition.in_arcs)[0].source)
      place_out_signature = self.get_place_signature(list(silent_transition.out_arcs)[0].target)
      silent_transition.correct = False
      number_of_nodes += 1
      for true_silent_transition in [t for t in petrinet_true.transitions if t.label is None]:
        true_place_in_signature = self.get_place_signature(list(true_silent_transition.in_arcs)[0].source)
        true_place_out_signature = self.get_place_signature(list(true_silent_transition.out_arcs)[0].target)
        if self.compare_place_signatures(place_in_signature, true_place_in_signature) and \
           self.compare_place_signatures(place_out_signature, true_place_out_signature):
          silent_transition.correct = True
          silent_transition.marked = True
          number_of_nodes_correct += 1
          break
      else:
        all_correct = False

    return all_correct, number_of_nodes_correct, number_of_nodes

  def get_statistics(self, places=None):
    counts = {'#': 0, 'exact': 0, 'under specified': 0, 'over specified': 0, 'under+over specified': 0}
    statistics = {'true positive': {**counts}, 'false positive': {**counts}, 'false negative': {**counts}}
    if places is None:
      places = self.petrinet.places
    for place in places:
      if place.correct:
        stat = statistics['true positive']
      else:
        if not hasattr(place, 'false_negative'):
          stat = statistics['false positive']
        elif place.false_negative:
          stat = statistics['false negative']

      stat['#'] += 1
      if len(place.evaluation.under_specified) > 0 and len(place.evaluation.over_specified) > 0:
        stat['under+over specified'] += 1
      elif len(place.evaluation.under_specified) > 0:
        stat['under specified'] += 1
      elif len(place.evaluation.over_specified) > 0:
        stat['over specified'] += 1
      else:
        stat['exact'] += 1
    return statistics


  def set_input_output_transitions(self, place):
    if not hasattr(place, 'input_transitions'):
      place.input_transitions = set([arc.source.label.replace('\n', ' ') for arc in place.in_arcs if arc.source.label is not None])
      place.output_transitions = set([arc.target.label.replace('\n', ' ') for arc in place.out_arcs if arc.target.label is not None])

  def analyze_places(self, alpha_relations, places=None):
    if places is None:
      places = self.petrinet.places
    for place in places:
      self.set_input_output_transitions(place)
      place.evaluation = PlaceEvaluation(place.input_transitions, place.output_transitions, alpha_relations=alpha_relations)
      place.evaluation.analyze_information()

  def print_analysis(self, places=None, verbosity=1):
    if places is None:
      places = self.petrinet.places
    try:
      sorted_places = sorted(places, key=lambda place: int(str(place)[1:]))
    except:
      sorted_places = sorted(places, key=lambda place: str(place))
    for place in sorted_places:
      print(f'{Fore.CYAN if place.correct else Fore.RED}{place} {place.input_transitions} {place.output_transitions}{Style.RESET_ALL}')
      if len(place.evaluation.under_specified) != 0 or len(place.evaluation.over_specified) != 0:
        if verbosity > 1:
          print(f'{"directly follows":<16} {place.evaluation.directly_follows}')
          print(f'{"causal":<16} {place.evaluation.causal_relations}')
          print(f'{"parallel":<16} {place.evaluation.parallel_relations}')
        if verbosity == 1:
          print('under specified', place.evaluation.under_specified)
          print('over specified', place.evaluation.over_specified)

  def get_alignment_fitness(self, log, timeout=None):
    if timeout is None:
      aligned_fitness = conformance.fitness_alignments(log, self.petrinet, self.initial_marking, self.final_marking)
      aligned_precision = conformance.precision_alignments(log, self.petrinet, self.initial_marking, self.final_marking)
      return aligned_fitness, aligned_precision

    def handler(signum, frame):
      raise TimeoutError('end of time')

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
      aligned_fitness = conformance.fitness_alignments(log, self.petrinet, self.initial_marking, self.final_marking)
      aligned_precision = conformance.precision_alignments(log, self.petrinet, self.initial_marking, self.final_marking)
      signal.alarm(0)
    except TimeoutError:
      print(f'Alignment time out ({timeout}s).')
      aligned_fitness = None
      aligned_precision = None
    print('fdsafdsa', aligned_fitness, aligned_precision)
    return aligned_fitness, aligned_precision

  def alignment_based_conformance(self, log, alignment_timeout=None):
    try:
      aligned_fitness, aligned_precision = self.get_alignment_fitness(log, timeout=alignment_timeout)
      return aligned_fitness, aligned_precision
    except Exception as e:
      print(e)
      print('Can\'t perform alignment based conformance checking, since the net is not a relaxed sound net.')
      return None, None

  def entropy_based_conformance(self, log, exact=False, temp_filename='', timeout=-1):
    entropia_basedir = '/mnt/c/Users/s140511/tue/thesis/codebase/jbpt-pm/entropia'
    entropia = Entropia(log, self.petrinet, self.initial_marking, self.final_marking, entropia_basedir, temp_filename=temp_filename)
    try:
      results = entropia.compute(exact=exact, timeout=timeout, verbose=False)
      return results['recall'], results['precision']
    except Exception as e:
      print(e)
      return None, None

  def conformance(self, log, alignment_based=False, alignment_timeout=None, entropy_based=False, temp_filename='', entropy_timeout=-1):
    results = evaluator.apply(log, self.petrinet, self.initial_marking, self.final_marking)
    results['fitness_percFitTraces'] = results['fitness']['perc_fit_traces']
    results['fitness'] = results['fitness']['log_fitness']
    for k in ['fitness_alignments', 'fitness_alignments_percFitTraces', 'precision_alignments', 'fscore_alignments',
              'fitness_entropy_partial', 'precision_entropy_partial', 'fscore_entropy_partial',
              'fitness_entropy_exact', 'precision_entropy_exact', 'fscore_entropy_exact']:
      results[k] = None
    if alignment_based:
      aligned_fitness, aligned_precision = self.alignment_based_conformance(log, alignment_timeout)
      print()
      print(aligned_fitness, aligned_precision)
      results['fitness_alignments'] = None if aligned_fitness is None else aligned_fitness['averageFitness']
      results['fitness_alignments_percFitTraces'] = None if aligned_fitness is None else aligned_fitness['percFitTraces']
      results['precision_alignments'] = aligned_precision
      results['fscore_alignments'] = self.compute_fscore(results['fitness_alignments'], results['precision_alignments'])
    if entropy_based:
      for exact in [False, True]:
        postfix = 'exact' if exact else 'partial'
        entropy_fitness, entropy_precision = self.entropy_based_conformance(log, exact=exact, temp_filename=temp_filename, timeout=entropy_timeout)
        results[f'fitness_entropy_{postfix}'] = entropy_fitness
        results[f'precision_entropy_{postfix}'] = entropy_precision
        results[f'fscore_entropy_{postfix}'] = self.compute_fscore(results[f'fitness_entropy_{postfix}'], results[f'precision_entropy_{postfix}'])
    return results

  def compute_fscore(self, fitness, precision):
    if fitness is None or precision is None:
      return None
    if fitness + precision == 0:
      return 0
    return (2 * fitness * precision) / (fitness + precision)

  def visualize(self):
    gviz = pn_visualizer.apply(self.petrinet, self.initial_marking, self.final_marking)
    pn_visualizer.view(gviz)

# def getBestFits(table, topX=1):
#   sorted_by_f = sorted(table[1:], key=lambda x: float(x[1]), reverse=True)
#
#   models_beam_search = {'gcn_candidates_frequency_ln_': 0, 'gcn_candidates_ln_': 0, 'gcn_chosen_ln_': 0, 'gcn_candidates_frequency_new_jp_': 0, 'gcn_candidates_frequency_new_ln_': 0,
#                         'gcn_candidates_frequency_jp_': 0, 'gcn_candidates_jp_': 0, 'gcn_chosen_jp_': 0, 'gcn_candidates_new_jp_': 0}
#
#   sorted_table = [[v for v in table[0]]]
#   for row in sorted_by_f:
#     skip = False
#     for model in models_beam_search.keys():
#       if model in row[0]:
#         if models_beam_search[model] >= topX:
#           skip = True
#         else:
#           models_beam_search[model] += 1
#     if not skip:
#       sorted_table.append(row)
#
#   return sorted_table


# def toPrettyTable(table):
#   ptable = PrettyTable(table[0])
#   for row in table[1:]:
#     ptable.add_row(row)
#   return ptable
#
#
# def perform_conformance_checking(log_filenames, log_directory, dataset_names, pnml_filenames=list, export=False, sort=True):
#   header = ['name', 'soundness', 'easy_soundness','fscore_alignments', 'fscore', 'fitness_alignments', 'fitness_alignments_percFitTraces', 'fitness', 'precision_alignments', 'precision', 'generalization', 'metricsAverageWeight', 'simplicity', '#p', '#st']
#   csv_table = [header]
#   if export:
#     if len(dataset_names) == 1:
#       conformance_filename = f'{log_directory}predictions/{dataset_names[0]}_conformance_final'
#     else:
#       conformance_filename = f'{log_directory}predictions/all_conformance_final'
#     csv_filename = f'{conformance_filename}.csv'
#
#     if not os.path.exists(csv_filename):
#       with open(csv_filename, 'w') as file:
#         writer = csv.writer(file, delimiter=',')
#         writer.writerow(header)
#
#   for log_filename, dataset_name in tqdm.tqdm(zip(log_filenames, dataset_names)):
#     log = xes_importer.apply(log_filename)
#     no_prediction = False
#     print(dataset_name)
#
#     for pnml_filename in pnml_filenames:
#       pnml_filename = f'{dataset_name}'.join(pnml_filename.split('<dataset>'))
#       idd = pnml_filename.split('/')[-1].split(f'{dataset_name}_')[-1].split('.')[0]
#       if idd == dataset_name:
#         idd = 'groundtruth'
#       try:
#         petrinet_handler = PetrinetHandler()
#         petrinet_handler.importFromFile(pnml_filename)
#
#         if 'split' in pnml_filename:
#           if len(petrinet_handler.mInitialMarking.keys()) == 0:
#             for place in petrinet_handler.mPetrinet.places:
#               if len(place.in_arcs) == 0:
#                 petrinet_handler.mInitialMarking[place] = 1
#           if len(petrinet_handler.mFinalMarking.keys()) == 0:
#             for place in petrinet_handler.mPetrinet.places:
#               if len(place.out_arcs) == 0:
#                 petrinet_handler.mFinalMarking[place] = 1
#         petrinet_handler.create_unique_start_place()
#
#       except OSError:
#         print(f'{Fore.BLUE + pnml_filename + Style.RESET_ALL} n/a')
#         row = [f'{dataset_name}/{idd}', 'False', 'False', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']
#         csv_table.append(row)
#         if export:
#           with open(csv_filename, 'a') as file:
#             writer = csv.writer(file, delimiter=',')
#             writer.writerow(row)
#         continue
#
#       petrinet_evaluation = PetrinetEvaluation(*petrinet_handler.get())
#
#       soundness      = petrinet_handler.get_pm4py_soundness(timeout=15)
#       easy_soundness = petrinet_handler.get_easy_soundness(timeout=15)
#       print(f'Sound: {Fore.GREEN if soundness[0] else Fore.RED}{soundness[0]}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
#       result = petrinet_evaluation.conformance(log, alignment_based=easy_soundness, alignment_timeout=30)
#
#       print(f'{Fore.BLUE + pnml_filename + Style.RESET_ALL} {result["fscore"]:.3f} {len(petrinet_handler.mPetrinet.places)} {len([t for t in petrinet_handler.mPetrinet.transitions if t.label is None])}')
#
#       row_temp = {}
#       if 'fitness_alignments' in result and 'precision_alignments' in result:
#         row_temp[f'{"fitness_alignments"}'] = f'{result["fitness_alignments"]["averageFitness"]:.5f}'
#         row_temp[f'{"fitness_alignments_percFitTraces"}'] = f'{result["fitness_alignments"]["percFitTraces"]:.5f}'
#         row_temp[f'{"precision_alignments"}'] = f'{result["precision_alignments"]:.5f}'
#         fscore = (2 * result["fitness_alignments"]["averageFitness"] * result["precision_alignments"]) / (result["fitness_alignments"]["averageFitness"] + result["precision_alignments"])
#         row_temp[f'{"fscore_alignments"}'] = f'{fscore:.5f}'
#
#       row = [
#         f'{dataset_name}/{idd}',
#         f'{soundness[0]}',
#         f'{easy_soundness}',
#         row_temp.get('fscore_alignments', 'n/a'),
#         f'{result["fscore"]:.5f}',
#         row_temp.get('fitness_alignments', 'n/a'),
#         row_temp.get('fitness_alignments_percFitTraces', 'n/a'),
#         f'{result["fitness"]["log_fitness"]:.5f}',
#         row_temp.get('precision_alignments', 'n/a'),
#         f'{result["precision"]:.5f}',
#         f'{result["generalization"]:.5f}',
#         f'{result["metricsAverageWeight"]:.5f}',
#         f'{result["simplicity"]:.5f}',
#         f'{len(petrinet_handler.mPetrinet.places):>2}',
#         f'{len([t for t in petrinet_handler.mPetrinet.transitions if t.label is None]):>2}',
#       ]
#       csv_table.append(row)
#       if export:
#         with open(csv_filename, 'a') as file:
#           writer = csv.writer(file, delimiter=',')
#           writer.writerow(row)
#
#   if sort:
#     best_table = getBestFits(csv_table, topX=1)
#   else:
#     best_table = csv_table
#
#   table = toPrettyTable(best_table)
#   if sort:
#     table.get_string(sortby=('fscore'), reversesort=True)
#   else:
#     table.get_string()
#   print(table)
#
#   if export and len(best_table) > 1:
#     with open(f'{conformance_filename}.txt', 'w') as file:
#       if sort:
#         file.write(table.get_string(sortby=('fscore'), reversesort=True))
#       else:
#         file.write(table.get_string())
