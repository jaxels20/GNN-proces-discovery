from project.data_handling.petrinet import PetrinetHandler
from project.data_handling.log import LogHandler
from project.ml_models.preprocessing import GraphBuilder

import tqdm
import os

what_to_do = 'synthetic_dataset'

#####################################################################################
#################################### PNML TO PNG ####################################
#####################################################################################
if what_to_do == 'pnml_to_png':
  base_dir = '/home/dominique/TUe/thesis/git_data/evaluation_data'
  datasets = [
    'road_traffic_fine',
    'sepsis',
    'BPI_2012_A',
    'BPI_2012_O',
    'BPI_2017_A',
    'BPI_2017_O',
    'BPI_2020_Domestic_declarations',
    'BPI_2020_International_declarations',
    'BPI_2020_Permit_log',
    'BPI_2020_Prepaid_travel_cost',
    'BPI_2020_Request_for_payment'
  ]

  algorithms = ['heuristics_reduced', 'inductive_reduced', 'ilp_reduced', 'ilp', 'split_reduced', 'split']

  for dataset in datasets:
    print(f'DATASET {dataset}')
    for algorithm in algorithms:
      print(f'ALGORITHM {algorithm}')
      filename = f'{base_dir}/{dataset}/predictions/data_{algorithm}'
      petrinet_handler = PetrinetHandler()

      petrinet_handler.importFromFile(f'{filename}.pnml')
      petrinet_handler.visualize(fExport=f'{filename}.png')


#####################################################################################
################################# SYNTHETIC DATASET #################################
#####################################################################################
if what_to_do == 'synthetic_dataset':
  data_directory = '/home/dominique/TUe/thesis/git_data/process_trees_medium_ws2'

  counter = 0
  max_ = 0
  distinct_pbs = set()
  for i in tqdm.tqdm(range(2729)):
    petrinet_handler = PetrinetHandler()
    petrinet_handler.importFromFile(f'{data_directory}/petrinets_original/{i:04d}.pnml')

    # petrinet_handler.label_silent_transitions(keepSafe=1)
    # while petrinet_handler.label_silent_transitions(keepSafe=2):
    #   continue

    # petrinet_handler.visualize()

    transitions = [t for t in petrinet_handler.mPetrinet.transitions if t.label is not None]
    if len(transitions) > 18:
      print('hay')
      continue

    petrinet_handler.addStartAndEndTransitions()

    # petrinet_handler.visualize()

    log_handler = LogHandler(None)
    log_handler._importVariants(f'{data_directory}/logs_compressed/{i:04d}.npz')

    topX = 30
    traces = [list(variant['variant']) for variant in log_handler.mVariants][:topX]
    counts = [variant['count'] for variant in log_handler.mVariants][:topX]
    transitionLabels = ['>'] + [t[0] for t in log_handler.mTransitions] + [None, '|']

    pb = GraphBuilder(i, 21, traces, counts, transitionLabels, fDepth=1, fPetrinetHandler=petrinet_handler, embedding_strategy='onehot')

    stringie = '--'.join(sorted(pb.mTargetPlaces))

    if stringie in distinct_pbs:
      print('EXISTS')
      continue

    distinct_pbs.add(stringie)

    os.system(f'cp {data_directory}/logs/{i:04d}.xes {data_directory}/logs_f/{counter:04d}.xes')
    os.system(f'cp {data_directory}/logs_compressed/{i:04d}.npz {data_directory}/logs_compressed_f/{counter:04d}.npz')

    os.system(f'cp {data_directory}/petrinets_original/{i:04d}.pnml {data_directory}/petrinets/{counter:04d}.pnml')
    os.system(f'cp {data_directory}/petrinets_original_png/{i:04d}.png {data_directory}/petrinets_png/{counter:04d}.png')

    counter += 1
