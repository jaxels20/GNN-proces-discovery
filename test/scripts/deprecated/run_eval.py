import os
import tqdm


datasets = [
  # ('BPI_2012_A', 17),
  # ('BPI_2012_O', 30),
  # ('BPI_2017_A', 30),
  # ('BPI_2017_O', 30),
  # ('BPI_2020_Domestic_declarations', 8),
  # ('BPI_2020_International_declarations', 3),
  # ('BPI_2020_Permit_log', 75),
  # ('BPI_2020_Prepaid_travel_cost', 30),
  ('BPI_2020_Request_for_payment', 8),
  ('road_traffic_fine', 30),
  ('sepsis', 8),
]


models = [
  ('/mnt/c/Users/s140511/tue/thesis/git/project/ml_models/models/simple_ws2_frequency_144_.pth', 'simpleep144'),
  ('/mnt/c/Users/s140511/tue/thesis/git/project/ml_models/models/complex_ws_frequency_100_.pth', 'complexep100')
]
for dataset, tx in tqdm.tqdm(datasets):
  for model, ex in tqdm.tqdm(models):
    print(dataset)
    print(model)
    '''
    python3 -m project.process_mining -l 2187,2333 -d /mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/logs/ -cc -m ai -mf /mnt/c/Users/s140511/tue/thesis/git/project/ml_models/models/simple_ws2_frequency_144_.pth -bw -10 -bl 1 -np 1 -ln 0 -e ep144b
    '''
    d = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data'
    eval_command = f'python3 -m project.process_mining -l data -d {d}/{dataset}/ -cc -m ai -ln 0 -tx {tx} -bw -50 -bl 20 -np 20 -mf {model} -e {ex}'

    os.system(eval_command)
