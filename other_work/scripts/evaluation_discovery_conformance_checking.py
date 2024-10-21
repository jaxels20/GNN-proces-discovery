import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str)
parser.add_argument('-md', '--model_dir', type=str)

args = parser.parse_args()
data_dir = args.data_dir   #'/mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data'
model_dir = args.model_dir #'/mnt/c/Users/s140511/tue/thesis/git_clean/project/ml_models/models'

mf_models = [
  ('model_candidates_frequency_new_036', {'export': 'medium'}),
  ('complex_ws_frequency_100_', {'e': 'complex'}),
  ('simple_ws2_frequency_144_', {'e': 'simple'})
]

evaluation_data = [
#  ('road_traffic_fine', {'top_x_traces': 30}),
# ('sepsis', {'top_x_traces': 8}),
#  ('BPI_2012_A', {'top_x_traces': 17}),
#  ('BPI_2012_O', {'top_x_traces': 30}),
#  ('BPI_2017_A', {'top_x_traces': 30}),
#  ('BPI_2017_O', {'top_x_traces': 30}),
#  ('BPI_2020_Domestic_declarations', {'top_x_traces': 8}),
#  ('BPI_2020_International_declarations', {'top_x_traces': 3}),
#  ('BPI_2020_Permit_log', {'top_x_traces': 5}),
#  ('BPI_2020_Prepaid_travel_cost', {'top_x_traces': 5}),
#  ('BPI_2020_Request_for_payment', {'top_x_traces': 8})
    ('2012_BPI', {'top_x_traces': 30})
]

base_args = '--logFilename data --miner gnn --length_normalization 0 --conformanceChecking'
base_args += ' --beam_width -50 -bl 20 --number_of_petrinets 20'

reproduce = 'gnn'

if reproduce == 'gnn':
  for model, model_kwargs in tqdm.tqdm(mf_models):
    model_args = '' # ' '.join(f'--{k} {v}' for k, v in model_kwargs.items())
    model_args += f' --model_filename {model_dir}/{model}'
    for dataset, dataset_kwargs in tqdm.tqdm(evaluation_data):
      dataset_args = ' '.join(f'--{k} {v}' for k, v in dataset_kwargs.items())
      dataset_args += f' --dataDirectory {data_dir}/{dataset}/'
      print(dataset)
      print(model)

      eval_command = f'python3 -m project.process_mining {base_args} {dataset_args} {model_args}'
      print(eval_command)
      os.system(eval_command)
else:
  for method in ['inductive', 'split', 'heuristics']:
    base_args = f'--logFilename data --miner import --conformanceChecking'
    model_args = f'--import_petrinet results/data_{method}_reduced.pnml --export {method}'
    for dataset, dataset_kwargs in tqdm.tqdm(evaluation_data):
      print(dataset)
      print(method)
      dataset_args = f' --dataDirectory {data_dir}/{dataset}/'

      eval_command = f'python3 -m project.process_mining {base_args} {dataset_args} {model_args}'
      os.system(eval_command)
