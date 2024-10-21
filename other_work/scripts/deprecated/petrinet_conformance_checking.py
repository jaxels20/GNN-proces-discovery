from project.evaluation.petrinets import perform_conformance_checking

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--datasetname', type=str)
parser.add_argument('-ld', '--log_directory', type=str)
parser.add_argument('-md', '--model_directory', type=str)
parser.add_argument('-e', '--export', help='Export the results', action='store_true')
args = parser.parse_args()

from pm4py.objects.petri_net.importer import importer as pnml_importer

results = []

if args.model_directory is None:
  args.model_directory = args.log_directory

if args.datasetname is None:
  datasets = range(2414, 2663)
  log_filenames = [f'{args.log_directory}{dataset_name:04d}.xes' for dataset_name in datasets]
  # logs = []
  # for log_filename in log_filenames:
  #   print(log_filename)
  #   logs.append(xes_importer.apply(log_filename))
  dataset_names = [f'{dataset_name:04d}' for dataset_name in datasets]
else:
  log_filenames = [f'{args.log_directory}{args.datasetname}.xes']
  # logs = [xes_importer.apply(log_filename)]
  dataset_names = [args.datasetname]

from pm4py.statistics.traces.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter

pnml_filenames = ['_alpha', '_inductive_reduced', '_heuristics_reduced', '_split_reduced', '_ilp_reduced',
                  *[f'_gcn_candidates_frequency_new_ln_{i}' for i in range(6)],
                  '_gcn_candidates_frequency_new_jp', *[f'_gcn_candidates_frequency_new_jp_{i}' for i in range(6)],
                  '_gcn_candidates_new_jp', *[f'_gcn_candidates_new_jp_{i}' for i in range(6)],
                  '_gcn_candidates_frequency_ln', *[f'_gcn_candidates_frequency_ln_{i}' for i in range(6)],
                  '_gcn_candidates_frequency_jp', *[f'_gcn_candidates_frequency_jp_{i}' for i in range(6)],
                  '_gcn_candidates_ln', *[f'_gcn_candidates_ln_{i}' for i in range(6)],
                  '_gcn_candidates_jp', *[f'_gcn_candidates_jp_{i}' for i in range(6)],
                  '_gcn_chosen_ln', *[f'_gcn_chosen_ln_{i}' for i in range(6)],
                  '_gcn_chosen_jp', *[f'_gcn_chosen_jp_{i}' for i in range(6)],
                  # '_gcn_full_ln', *[f'_gcn_full_ln_{i}' for i in range(6)],
                  # '_gcn_full_jp', *[f'_gcn_full_jp_{i}' for i in range(6)],
                 ]

pnml_filenames = [f'{args.model_directory}../petrinets/<dataset>.pnml',
                  f'{args.model_directory}predictions/<dataset>_gcn_sound.pnml',
                  f'{args.log_directory}predictions/<dataset>_ilp.pnml',
                  f'{args.log_directory}predictions/<dataset>_split.pnml',
                  f'{args.log_directory}predictions/<dataset>_heuristics.pnml',
                  f'{args.log_directory}predictions/<dataset>_inductive.pnml']

# pnml_filenames = [f'_inductive']


perform_conformance_checking(log_filenames, args.log_directory, dataset_names, pnml_filenames=pnml_filenames, export=args.export, sort=False)