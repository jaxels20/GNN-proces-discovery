import pm4py

from project.data_handling.log import LogHandler
from project.data_handling.petrinet import PetrinetHandler
from project.ml_models.inference import Inference
from project.ml_models.model_generative import load_from_file
from project.ml_models.preprocessing import GraphBuilder
from project.ml_models.utils import print_variants
from project.evaluation.petrinets import PetrinetEvaluation

from pm4py.algo.discovery.alpha      import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive  import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net    import visualizer as vis_factory

from prompy import DocumentationGenerator
# DocumentationGenerator.generate_documentation('Documentation.py', parameters)
from prompy import ProMExecutor, ScriptBuilder

import numpy as np
from pprint import pprint
import json
from colorama import Fore, Style
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

class ProcessDiscovery:
  def __init__(self, fLogFilename):
    self.mLogHandler = LogHandler(fLogFilename)
    self.mNet = None
    self.mInitialMarking = None
    self.mFinalMarking = None

  def visualize(self):
    gviz = vis_factory.apply(self.mNet, self.mInitialMarking, self.mFinalMarking)
    vis_factory.view(gviz)

  def discover(self):
    raise NotImplementedError('Method to be implemented in concrete class iso this abstract class.')

  def export(self, fPetrinetName, fPetrinetNamePNG=None, fDebug=False):
    if fPetrinetNamePNG is None:
      fPetrinetNamePNG = fPetrinetName
    print(fPetrinetName)
    if self.mNet is not None:
      petrinet_handler = PetrinetHandler()
      petrinet_handler.mPetrinet = self.mNet
      petrinet_handler.mInitialMarking = self.mInitialMarking
      petrinet_handler.mFinalMarking = self.mFinalMarking

      petrinet_handler.merge_initial_final_marking()

      petrinet_handler.export(f'{fPetrinetName}.pnml')
      petrinet_handler.visualize(fDebug=fDebug, fExport=f'{fPetrinetNamePNG}.png')

  def soundness(self):
    petrinet_handler = PetrinetHandler()
    petrinet_handler.mPetrinet, petrinet_handler.mInitialMarking, petrinet_handler.mFinalMarking = self.mNet, self.mInitialMarking, self.mFinalMarking
    soundness, soundness_info = petrinet_handler.get_pm4py_soundness()
    easy_soundness = True if soundness else petrinet_handler.get_easy_soundness(timeout=15)
    print(f'Sound: {Fore.GREEN if soundness else Fore.RED}{soundness}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
    return (soundness, soundness_info), easy_soundness

  def conformance(self, log, soundness=None, easy_soundness=None, temp_filename=''):
    if soundness is None or easy_soundness is None:
      (soundness, _), easy_soundness = self.soundness()
    petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)

    return petrinet_evaluation.conformance(log, alignment_based=easy_soundness, alignment_timeout=300,
                                           entropy_based=easy_soundness, temp_filename=temp_filename, entropy_timeout=300)

  def conformance_export(self, filename, petrinet_filename, log_filename, soundness, soundness_info, easy_soundness,
                         result, result_top_x=None, topXTraces=0, args=None):
    args = {} if args is None else args
    soundness_info_output = {
      's_components': [[str(el) for el in comp] for comp in soundness_info.get('s_components', [])],
      'uncovered_places_s_component': [str(el) for el in soundness_info.get('uncovered_places_s_component', [])]}
    conformance_output = {'petrinet_filename': petrinet_filename, 'log_filename': log_filename,
                          'L': result, f'L_{topXTraces}': result_top_x, 'sound': soundness,
                          'easy_soundness': easy_soundness, 'soundness_info': soundness_info_output, 'args': args}
    with open(filename, 'w') as cc_file:
      json.dump(conformance_output, cc_file, sort_keys=True, indent=2)
    return conformance_output

  def print_conformance(self, result):
    parse_value = lambda x: f'{x:.5f}' if isinstance(x, int) or isinstance(x, float) else f'{x}'
    print('\n'.join([f'{k:<32}: {parse_value(v)}' for k, v in sorted(result.items())]))


class GnnMiner(ProcessDiscovery):
  def __init__(self, fLogFilename, model_filename, embedding_size=21, embedding_strategy='onehot', include_frequency=True):
    super().__init__(None)
    self.log_filename = fLogFilename
    self.embedding_size = embedding_size
    self.embedding_strategy = embedding_strategy
    self.include_frequency = include_frequency #('frequency' in model_filename.split('/')[-1])
    self.model = load_from_file(model_filename, embedding_size, include_frequency=self.include_frequency)
    self.model_inference = Inference(self.model)

  def discover(self, export='', conformance_check=True, beam_width=1, beam_length=1, number_of_petrinets=1,  topXTraces=None,
               length_normalization=True, timeout=None, args=None):
    self.args = {} if args is None else args
    try:
      self.mLogHandler._importVariants(self.log_filename)
    except FileNotFoundError:
      if 'logs' in self.log_filename:
        self.mLogHandler._importVariants(self.log_filename.replace('logs', 'logs_compressed'))
      else:
        raise FileNotFoundError

    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]

    from pm4py.objects.log.importer.xes import importer as xes_importer

    if topXTraces is None:
      percentage = 80
      variants = self.mLogHandler.getMostFrequentVariants(percentage, minimum_variants=30, maximum_variants=75)
      topXTraces = len(variants)
    else:
      variants = self.mLogHandler.getMostFrequentVariants(100, minimum_variants=topXTraces, maximum_variants=topXTraces)

    traces = [list(variant['variant']) for variant in variants]
    counts = [variant['count'] for variant in variants]
    print(f'{len(self.mLogHandler.mVariants)} variants in original log, taking {len(traces)}.')

    labels = [t for t in self.mLogHandler.mTransitions.keys()]

    show_log = True
    if show_log:
      log_filename = f'{self.log_filename}.xes'
      log = xes_importer.apply(log_filename)

      from pm4py.statistics.traces.log import case_statistics
      from pm4py.algo.filtering.log.variants import variants_filter

      variants_count = case_statistics.get_variant_statistics(log)
      variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
      traces_ = [variant['variant'] for variant in variants_count][:len(traces)]
      counts_ = [variant['count'] for variant in variants_count][:len(traces)]
      print_variants(traces_, labels, counts_)

    if conformance_check:
      log_filename = f'{self.log_filename}.xes'
      log = xes_importer.apply(log_filename)

      from pm4py.statistics.traces.log import case_statistics
      from pm4py.algo.filtering.log.variants import variants_filter

      variants_count = case_statistics.get_variant_statistics(log)
      variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
      variants = [variant['variant'] for variant in variants_count][:topXTraces]
      log_top_x = variants_filter.apply(log, variants)

    transitionLabels = ['>'] + labels + ['|']
    if len(transitionLabels) > self.embedding_size:
      # TODO improve logging.
      print('=' * 60, 'too many transitions')
      return
    name = self.mLogHandler.mLogFilename.split('/')[-1].split('.')[0]

    start_construction_time = time.time()
    
    # HERE THEY BUILD THE GRAPH 
    pb = GraphBuilder(name, self.embedding_size, traces, counts, transitionLabels, fDepth=1, embedding_strategy=self.embedding_strategy, include_frequency=self.include_frequency, fPetrinetHandler=None)
    print(f'Graph construction took: {time.time() - start_construction_time:.3f} seconds')
    print('number of nodes', pb.mNet.number_of_nodes())
    print('number of candidate places', len(pb.mPossiblePlaces))
    beam_width = min(len(pb.mPossiblePlaces), beam_width)
    start_inference_time = time.time()

    results = self.model_inference.predict(pb, beam_width=beam_width, beam_length=beam_length, max_number_of_places=50,
                                           number_of_petrinets=number_of_petrinets, length_normalization=length_normalization,
                                           transitionLabels=transitionLabels, timeout=timeout)
    print(f'Inference took: {time.time() - start_inference_time:.3f} seconds')

    best_conformance = None
    petrinets_ids = []
    results = results[:number_of_petrinets]

    conformance_outputs = {}
    conformance_output_filename = f'{data_directory}/predictions/{petrinet_name}_gcn_{export}'
    export_fn = f'{data_directory}/predictions/{petrinet_name}_gcn_{export}'
    export_fn_png = f'{data_directory}/predictions/pngs/{petrinet_name}_gcn_{export}'
    for i, result in enumerate(results):
      petrinet_handler = PetrinetHandler()
      print(f'Discovered {len(result["places"])} places and {len(result["silent_transitions"])} silent transitions '
            f'({len(result["places"]) + len(result["silent_transitions"])})')

      petrinet_id = (set(result['places']), set(result['silent_transitions']))
      if petrinet_id in petrinets_ids:
        print('Found this exact petri net already.')
        continue
      petrinets_ids.append(petrinet_id)
      petrinet_handler.fromPlaces(result['places'], transitionLabels, None, fSilentTransitions=result['silent_transitions'])
      petrinet_handler.move_initial_final_markings()
      petrinet_handler.remove_duplicate_silent_transitions()
      petrinet_handler.remove_loose_transitions()

      self.mNet, self.mInitialMarking, self.mFinalMarking = petrinet_handler.get()
      if export != '':
        prob = f'{result["probability"]:.4f}'.split('.')[-1]
        postfix = '' if len(results) == 1 else f'_{i}'
        self.export(f'{export_fn}{postfix}', f'{export_fn_png}{postfix}')
        self.export(f'{export_fn}{postfix}', f'{export_fn_png}{postfix}_{prob}', fDebug=True)

      if conformance_check:
        (soundness, soundness_info), easy_soundness = self.soundness()
        # petrinet_handler.visualize()

        print(Fore.MAGENTA, i, Style.RESET_ALL)
        result = self.conformance(log, soundness=soundness, easy_soundness=easy_soundness, temp_filename=export)
        print('On full log:')
        self.print_conformance(result)
        result_top_x = self.conformance(log_top_x, soundness=soundness, easy_soundness=easy_soundness, temp_filename=export)
        print(f'Only on top {topXTraces} trace variants:')
        self.print_conformance(result_top_x)

        score = float(soundness) + (result['fscore'] if result['fscore_alignments'] is None else result['fscore_alignments'])
        if best_conformance is None or score > best_conformance[2]:
          best_conformance = result, i, score

        if export != '':
          postfix = '' if len(results) == 1 else f'_{i}'
          conformance_output = self.conformance_export(f'{conformance_output_filename}{postfix}_cc.json', f'{export_fn}{postfix}',
                                                       self.log_filename, soundness, soundness_info, easy_soundness,
                                                       result, result_top_x, topXTraces, args)
          conformance_outputs[i] = {k: v for k, v in sorted(conformance_output.items())}

    if best_conformance is not None:
      print(Fore.MAGENTA, 'best conformance:', best_conformance[1], Style.RESET_ALL)
      self.print_conformance(best_conformance[0])

    if conformance_check and export != '' and len(results) > 1:
      # Export all conformance outputs together sorted on score.
      score_comp = lambda ir: int(ir[1]['sound']) + (ir[1]['L']['fscore'] if ir[1]['L']['fscore_alignments'] is None else ir[1]['L']['fscore_alignments'])
      data = [{'i': i, **conformance_output} for i, conformance_output in sorted(conformance_outputs.items(), key=score_comp, reverse=True)]
      with open(f'{conformance_output_filename}_all_cc.json', 'w') as cc_file:
        json.dump(data, cc_file, sort_keys=False, indent=2)
      with open(f'{"results".join(conformance_output_filename.split("predictions"))}_all_cc.json', 'w') as cc_file:
        print(data)
        json.dump(data, cc_file, sort_keys=False, indent=2)

    # # TODO remove
    # # Translate place-sets to petrinets
    # for i, result in enumerate(results):
    #   petrinet_handler = PetrinetHandler()
    #   print(f'Discovered {len(result["places"])} places and {len(result["silent_transitions"])} silent transitions '
    #         f'({len(result["places"]) + len(result["silent_transitions"])})')
    #   order_labels = [str(i) for i in range(len(result['places']))]
    #   choice_prob_labels = [f'{v.tolist():.2f}' for v in result['choice_probabilities']]
    #   stop_prob_labels = [f'{v.tolist()[0][0]:.2f}' for v in result['add_probabilities']]
    #
    #   print(result['places'])
    #   print(result['silent_transitions'])
    #   petrinet_id = (set(result['places']), set(result['silent_transitions']))
    #   if petrinet_id in petrinets_ids:
    #     print('Found this exact petri net already.')
    #     continue
    #   petrinets_ids.append(petrinet_id)
    #
    #   petrinet_handler.fromPlaces(result['places'], transitionLabels, None, fSilentTransitions=result['silent_transitions'])
    #
    #   petrinet_handler.move_initial_final_markings()
    #
    #   petrinet_handler.remove_duplicate_silent_transitions()
    #
    #   petrinet_handler.remove_loose_transitions()
    #   print('#t', len(petrinet_handler.mPetrinet.transitions))
    #   self.mNet, self.mInitialMarking, self.mFinalMarking = petrinet_handler.get()
    #
    #   if export != '':
    #     export_fn = f'{data_directory}/predictions/{petrinet_name}_gcn_{export}'
    #     export_fn_png = f'{data_directory}/predictions/pngs/{petrinet_name}_gcn_{export}'
    #     prob = f'{result["probability"]:.4f}'.split('.')[-1]
    #     if len(results) == 1:
    #       self.export(export_fn, export_fn_png)
    #       self.export(export_fn, f'{export_fn_png}_{prob}', fDebug=True)
    #     else:
    #       self.export(f'{export_fn}_{i}', f'{export_fn_png}_{i}')
    #       self.export(f'{export_fn}_{i}', f'{export_fn_png}_{i}_{prob}', fDebug=True)
    #
    #   if conformance_check:
    #     petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)
    #
    #     soundness, soundness_info = petrinet_handler.get_pm4py_soundness()
    #     if soundness:
    #       easy_soundness = True
    #     else:
    #       easy_soundness = petrinet_handler.get_easy_soundness(timeout=15)
    #
    #     print(Fore.MAGENTA, i, Style.RESET_ALL)
    #     print(f'Sound: {Fore.GREEN if soundness else Fore.RED}{soundness}{Style.RESET_ALL} easy sound {Fore.GREEN if easy_soundness else Fore.RED}{easy_soundness}{Style.RESET_ALL}')
    #     result = petrinet_evaluation.conformance(log, alignment_based=easy_soundness, entropy_based=soundness)
    #     print('On full log:')
    #     pprint(self.print_conformance(result))
    #     result_top_x = petrinet_evaluation.conformance(log_top_x, alignment_based=easy_soundness, entropy_based=soundness)
    #     print(f'Only on top {topXTraces} trace variants:')
    #     pprint(self.print_conformance(result_top_x))
    #
    #     # TODO fix search for best conformance, ie fscore_alignments over fscore and extra points for soundness.
    #
    #     aligned_fscore = None
    #
    #     if 'fitness_alignments' in result and 'precision_alignments' in result and result["precision_alignments"] > 0 and result["fitness_alignments"]["averageFitness"] > 0:
    #       aligned_fscore = (2 * result["fitness_alignments"]["averageFitness"] * result["precision_alignments"]) / (result["fitness_alignments"]["averageFitness"] + result["precision_alignments"])
    #     score = float(soundness) + (result['fscore'] if aligned_fscore is None else aligned_fscore)
    #     if best_conformance is None or score > best_conformance[2]:
    #       best_conformance = result, i, score
    #
    #     # if export != '':
    #     #   if len(results) > 1:
    #     #     conformance_filename = f'{dddd}/predictions/{petrinet_name}_gcn_{export}_{i}_cc.json'
    #     #     with open(conformance_filename, 'w') as cc_file:
    #     #       conformances = {'full': result, f'top {topXTraces}': result_top_x,
    #     #                       'sound': soundness, 's_components': [[str(el) for el in comp] for comp in soundness_info.get('s_components', [])],
    #     #                       'uncovered_places_s_component': [str(el) for el in soundness_info.get('uncovered_places_s_component', [])],
    #     #                       'easy_soundness': easy_soundness, 'filename': conformance_filename,
    #     #                       'args': args}
    #     #       json.dump(conformances, cc_file, sort_keys=True, indent=2)
    #     #   else:
    #     #     conformance_filename = f'{dddd}/predictions/{petrinet_name}_gcn_{export}_cc.json'
    #     #     with open(conformance_filename, 'w') as cc_file:
    #     #       conformances = {'full': result, f'top {topXTraces}': result_top_x,
    #     #                       'sound': soundness, 's_components': [[str(el) for el in comp] for comp in soundness_info.get('s_components', [])],
    #     #                       'uncovered_places_s_component': [str(el) for el in soundness_info.get('uncovered_places_s_component', [])],
    #     #                       'easy_soundness': easy_soundness, 'filename': conformance_filename,
    #     #                       'args': args}
    #     #       json.dump(conformances, cc_file, sort_keys=True, indent=2)
    #
    # # TODO Export combined conformance stats for each result, sorted on fscore_alignments.
    #
    # if best_conformance is not None:
    #   print(Fore.MAGENTA, 'best conformance:', best_conformance[1], Style.RESET_ALL)
    #   pprint(self.print_conformance(best_conformance[0]))


class AlphaMiner(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename

  def discover(self, export='', conformance_check=True, args=None):
    self.mNet, self.mInitialMarking, self.mFinalMarking = alpha_miner.apply(self.mLogHandler.mLog)
    dddd = '/'.join(self.log_filename.split('/')[:-1])
    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    if export != '':
      self.export(f'{dddd}/predictions/{petrinet_name}_alpha')


class HeuristicsMiner(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename

  def discover(self, conformance_check=False, export='', args=None):
    self.mNet, self.mInitialMarking, self.mFinalMarking = heuristics_miner.apply(self.mLogHandler.mLog)
    # petrinet_handler = PetrinetHandler()
    # petrinet_handler.mPetrinet, petrinet_handler.mInitialMarking, petrinet_handler.mFinalMarking = self.mNet, self.mInitialMarking, self.mFinalMarking
    # petrinet_handler.visualize()

    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    conformance_output_filename = f'{data_directory}/predictions/{petrinet_name}_heuristics'
    export_fn = f'{data_directory}/predictions/{petrinet_name}_heuristics'
    export_fn_png = f'{data_directory}/predictions/pngs/{petrinet_name}_heuristics'

    if export != '':
      self.export(export_fn, export_fn_png)

    if conformance_check:
      (soundness, soundness_info), easy_soundness = self.soundness()
      result = self.conformance(self.mLogHandler.mLog, soundness, easy_soundness, temp_filename=export)
      print('FULL:')
      pprint(self.print_conformance(result))
      if export != '':
        self.conformance_export(f'{conformance_output_filename}_cc.json', export_fn, self.log_filename,
                                soundness, soundness_info, easy_soundness, result)

    # if conformance_check:
    #   self.conformance()
    #   petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)
    #   result = petrinet_evaluation.conformance(self.mLogHandler.mLog, alignment_based=True)
    #   print('FULL:')
    #   pprint(self.print_conformance(result))
    #   if export != '':
    #     with open(f'{dddd}/predictions/{petrinet_name}_heuristics_cc.json', 'w') as cc_file:
    #       json.dump({'FULL': result}, cc_file, sort_keys=True, indent=2)


class InductiveMiner(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename

  def discover_prom(self, export_fn):
    # TODO no hardcoded paths.
    parameters = {
      'prom_directory': Path('/mnt/c/Users/s140511/tue/Courses/CapitaSelecta/prom'),  # Directory to the ProM installation.
      'lib_directory': Path('lib'),  # Lib directory, to be contained in the promDirectory, specified above.
      'dist_directory': Path('dist'),  # Dist directory, to be contained in the promDirectory, specified above.
      'memory': '4G',  # Memory for Java to use.
      'java': 'java'  # Java command: 'java' for Linux, 'jre8\\bin\\java' for Windows.
    }
    prom_executor = ProMExecutor.ProMExecutor(parameters)

    script = ScriptBuilder.mine(self.log_filename, 'inductive', {'parameters': 'imf'})
    script += ScriptBuilder.export_petrinet('petrinet', export_fn)
    script += ScriptBuilder.end()

    output = prom_executor.run_script(script, timeout=25, verbosity=0)
    print(output)

  def discover(self, conformance_check=False, export='', args=None):
    args = {} if args is None else args

    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    conformance_output_filename = f'{data_directory}/predictions/{petrinet_name}_inductive'
    export_fn = f'{data_directory}/predictions/{petrinet_name}_inductive'
    export_fn_png = f'{data_directory}/predictions/pngs/{petrinet_name}_inductive'

    if args.get('prom', False):
      self.discover_prom(export_fn)
      handler = PetrinetHandler()
      handler.importFromFile(export_fn)
      self.mNet, self.mInitialMarking, self.mFinalMarking = handler.get()
    else:
      self.mNet, self.mInitialMarking, self.mFinalMarking = inductive_miner.apply(self.mLogHandler.mLog, variant=inductive_miner.IMf)

    if export != '':
      self.export(export_fn, export_fn_png)

    if conformance_check:
      (soundness, soundness_info), easy_soundness = self.soundness()
      result = self.conformance(self.mLogHandler.mLog, soundness, easy_soundness, temp_filename=export)
      print('FULL:')
      pprint(self.print_conformance(result))
      if export != '':
        print(f'{conformance_output_filename}_cc.json')
        self.conformance_export(f'{conformance_output_filename}_cc.json', export_fn, self.log_filename,
                                soundness, soundness_info, easy_soundness, result)

    # dddd = '/'.join(self.log_filename.split('/')[:-1])
    # if 'logs' in dddd:
    #   dddd = '/'.join(dddd.split('/')[:-1])
    # petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    #
    # if export != '':
    #   self.export(f'{dddd}/predictions/{petrinet_name}_inductive', f'{dddd}/predictions/pngs/{petrinet_name}_inductive')
    #
    # if conformance_check:
    #   petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)
    #   result = petrinet_evaluation.conformance(self.mLogHandler.mLog, alignment_based=True)
    #   print('FULL:')
    #   pprint(self.print_conformance(result))
    #   if export != '':
    #     with open(f'{dddd}/predictions/{petrinet_name}_inductive_cc.json', 'w') as cc_file:
    #       json.dump({'FULL': result}, cc_file, sort_keys=True, indent=2)


class SplitMiner(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename
    self.split_miner_dir = '/mnt/c/Users/s140511/tue/thesis/splitminer'

  def discover(self, conformance_check=False, export='', args=None):
    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    conformance_output_filename = f'{data_directory}/predictions/{petrinet_name}_split'
    export_fn = f'{data_directory}/predictions/{petrinet_name}_split'
    export_fn_png = f'{data_directory}/predictions/pngs/{petrinet_name}_split'


    # dddd = '/'.join(self.log_filename.split('/')[:-1])
    # if 'logs' in dddd:
    #   dddd = '/'.join(dddd.split('/')[:-1])
    # petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    # petrinet_fn = f'{dddd}/predictions/{petrinet_name}_split'

    split_miner_command = f'java -cp splitminer.jar:./lib/\* au.edu.unimelb.services.ServiceProvider SMPN 0.1 0.4 true {self.log_filename}.xes {export_fn}'

    os.system(f'cd {self.split_miner_dir} && {split_miner_command} && cd -')

    petrinet_handler = PetrinetHandler()
    petrinet_handler.importFromFile(export_fn)
    for place in petrinet_handler.mPetrinet.places:
      if len(place.in_arcs) == 0:
        petrinet_handler.mInitialMarking[place] = 1
    self.mNet, self.mInitialMarking, self.mFinalMarking = petrinet_handler.get()
    if export != '':
      self.export(export_fn, export_fn_png)

    if conformance_check:
      (soundness, soundness_info), easy_soundness = self.soundness()
      result = self.conformance(self.mLogHandler.mLog, soundness, easy_soundness, temp_filename=export)
      print('FULL:')
      pprint(self.print_conformance(result))
      if export != '':
        self.conformance_export(f'{conformance_output_filename}_cc.json', export_fn, self.log_filename,
                                soundness, soundness_info, easy_soundness, result)

    # if export != '':
    #   self.export(f'{dddd}/predictions/{petrinet_name}_split', f'{dddd}/predictions/pngs/{petrinet_name}_split')
    #
    # if conformance_check:
    #   petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)
    #   result = petrinet_evaluation.conformance(self.mLogHandler.mLog, alignment_based=True)
    #   print('FULL:')
    #   pprint(self.print_conformance(result))
    #   if export != '':
    #     with open(f'{dddd}/predictions/{petrinet_name}_split_cc.json', 'w') as cc_file:
    #       json.dump({'FULL': result}, cc_file, sort_keys=True, indent=2)


class ImportMiner(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename

  def discover(self, conformance_check=False, export='', args=None):
    args = {} if args is None else args

    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    if (import_petrinet := args.get('import_petrinet', None)) is not None:
      export_fn = f'{data_directory}/{import_petrinet}'
      handler = PetrinetHandler()
      handler.importFromFile(export_fn)
      handler.set_initial_and_final_markings_when_empty()

      if len(handler.mInitialMarking) > 1 or len([place for place in handler.mPetrinet.places if len(place.in_arcs) == 0]) == 0:
        handler.create_unique_start_place()
      self.mNet, self.mInitialMarking, self.mFinalMarking = handler.get()
      conformance_output_filename = f'{data_directory}/{"".join(import_petrinet.split(".pnml"))}'
      print(conformance_output_filename)
    else:
      raise ValueError('Import miner needs --import_petrinet to be specified.')

    if conformance_check:
      (soundness, soundness_info), easy_soundness = self.soundness()
      result = self.conformance(self.mLogHandler.mLog, soundness, easy_soundness, temp_filename=export)
      print('FULL:')
      pprint(self.print_conformance(result))
      if export != '':
        print(f'{conformance_output_filename}_cc.json')
        self.conformance_export(f'{conformance_output_filename}_cc.json', export_fn, self.log_filename,
                                soundness, soundness_info, easy_soundness, result)


class GroundTruth(ProcessDiscovery):
  def __init__(self, fLogFilename):
    super().__init__(fLogFilename)
    self.log_filename = fLogFilename

  def discover(self, conformance_check=False, export='', args=None):
    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

    petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    conformance_output_filename = f'{data_directory}/predictions/{petrinet_name}_inductive'
    export_fn = f'{data_directory}/predictions/{petrinet_name}.pnml'
    # dddd = '/'.join(self.log_filename.split('/')[:-1])
    # if 'logs' in dddd:
    #   dddd = '/'.join(dddd.split('/')[:-1])
    # petrinet_name = self.log_filename.split('/')[-1].split('.')[0]
    # petrinet_fn = f'{dddd}/petrinets/{petrinet_name}.pnml'

    petrinet_handler = PetrinetHandler()
    petrinet_handler.importFromFile(export_fn)
    self.mNet, self.mInitialMarking, self.mFinalMarking = petrinet_handler.get()

    if conformance_check:
      (soundness, soundness_info), easy_soundness = self.soundness()
      result = self.conformance(self.mLogHandler.mLog, soundness, easy_soundness, temp_filename=export)
      print('FULL:')
      pprint(self.print_conformance(result))
      if export != '':
        self.conformance_export(f'{conformance_output_filename}_cc.json', export_fn, self.log_filename,
                                soundness, soundness_info, easy_soundness, result)

    # if conformance_check:
    #   petrinet_evaluation = PetrinetEvaluation(self.mNet, self.mInitialMarking, self.mFinalMarking)
    #   result = petrinet_evaluation.conformance(self.mLogHandler.mLog, alignment_based=True)
    #   print('FULL:')
    #   pprint(self.print_conformance(result))
    #   if export != '':
    #     with open(f'{dddd}/predictions/{petrinet_name}_groundtruth_cc.json', 'w') as cc_file:
    #       json.dump({'FULL': result}, cc_file, sort_keys=True, indent=2)