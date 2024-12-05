from gnn_miner.data_handling.log import LogHandler
from gnn_miner.data_handling.petrinet import PetrinetHandler
from gnn_miner.ml_models.inference import Inference
from gnn_miner.ml_models.model_generative import load_from_file
from gnn_miner.ml_models.preprocessing import GraphBuilder
from gnn_miner.ml_models.utils import print_variants
from pm4py.visualization.petri_net    import visualizer as vis_factory

import time

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

  def export(self, save_dir, fPetrinetName, fPetrinetNamePNG=None, fDebug=False):
    if fPetrinetNamePNG is None:
      fPetrinetNamePNG = fPetrinetName
    print(fPetrinetName)
    if self.mNet is not None:
      petrinet_handler = PetrinetHandler()
      petrinet_handler.mPetrinet = self.mNet
      petrinet_handler.mInitialMarking = self.mInitialMarking
      petrinet_handler.mFinalMarking = self.mFinalMarking

      petrinet_handler.merge_initial_final_marking()

      petrinet_handler.export(f'{save_dir}/{fPetrinetName}.pnml')
      petrinet_handler.visualize(fDebug=fDebug, fExport=f'{save_dir}/{fPetrinetNamePNG}.png')


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

    data_directory = '/'.join(self.log_filename.split('/')[:-1])
    if 'logs' in data_directory:
      data_directory = '/'.join(data_directory.split('/')[:-1])

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

    show_log = False #
    if show_log:
      log_filename = f'{self.log_filename}.xes'
      log = xes_importer.apply(log_filename)

      # from pm4py.statistics.traces.log import case_statistics
      from pm4py.statistics.traces.generic.log import case_statistics

      variants_count = case_statistics.get_variant_statistics(log)
      variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
      traces_ = [variant['variant'] for variant in variants_count][:len(traces)]
      counts_ = [variant['count'] for variant in variants_count][:len(traces)]
      print_variants(traces_, labels, counts_)

    transitionLabels = ['>'] + labels + ['|']
    if len(transitionLabels) > self.embedding_size:
      print('=' * 60, 'too many transitions')
      return
    name = self.mLogHandler.mLogFilename.split('/')[-1].split('.')[0]

    start_construction_time = time.time()
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
    
    petrinets_ids = []
    results = results[:number_of_petrinets]
    
    for _, result in enumerate(results):
      petrinet_id = (set(result['places']), set(result['silent_transitions']))
      if petrinet_id in petrinets_ids:
        print('Found this exact petri net already.')
        continue
      
      petrinet_handler = PetrinetHandler()
      print(f'Discovered {len(result["places"])} places and {len(result["silent_transitions"])} silent transitions '
            f'({len(result["places"]) + len(result["silent_transitions"])})')
      
      petrinets_ids.append(petrinet_id)
      petrinet_handler.fromPlaces(result['places'], transitionLabels, None, fSilentTransitions=result['silent_transitions'])
      petrinet_handler.move_initial_final_markings()
      petrinet_handler.remove_duplicate_silent_transitions()
      petrinet_handler.remove_loose_transitions()
      self.mNet, self.mInitialMarking, self.mFinalMarking = petrinet_handler.get()
