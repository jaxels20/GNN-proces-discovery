from project.evaluation.entropia import Entropia

def find_gcn_model(dataset, datadir2='evaluation_data'):
  datadir = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/{datadir2}'
  dir = f'{datadir}/{dataset}/results'
  gcns = []
  for filename in os.listdir(dir):
    if filename[-5:] == '.pnml' and 'gcn' in filename and 'temp' not in filename:
      gcns.append(filename.split('data_')[1].split('.pnml')[0])
  if len(gcns) != 1:
    print('waaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaah', gcns)
  return gcns[0]


def get_name(dataset, method):
  basedir = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_medium_ws2'
  if method == 'gcn_sound':
    return f'{basedir}/logs_compressed/predictions/{dataset}_gcn_sound.pnml'
  elif method == 'groundtruth':
    return f'{basedir}/petrinets/{dataset}.pnml'
  else:
    return f'{basedir}/logs/predictions/{dataset}_{method}.pnml'

def eval_data():
  # datadir = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data/'
  datasets = ['BPI_2012_A', 'BPI_2012_O', 'BPI_2017_A', 'BPI_2017_O', 'BPI_2020_Domestic_declarations',
              'BPI_2020_International_declarations', 'BPI_2020_Permit_log', 'BPI_2020_Prepaid_travel_cost',
              'BPI_2020_Request_for_payment', 'road_traffic_fine', 'sepsis']
  datasets = ['BPI_2020_Prepaid_travel_cost', 'BPI_2020_Request_for_payment', 'road_traffic_fine', 'sepsis']
  methods = ['split_reduced', 'gcn', 'inductive_reduced', 'heuristics_reduced', 'ilp_reduced']
  for dataset in datasets:
    print(dataset)
    for method in methods:
      if method == 'gcn':
        method = find_gcn_model(dataset)

      entropia = Entropia(dataset, method)
      entropia.compute(exact=False, verbose=True)
      entropia.export()
      print(method, entropia.results)

def test_data(datadir, samples):
  datasets = [f'{i:04d}' for i in samples]
  methods = ['groundtruth', 'gcn_sound', 'ilp', 'split', 'heuristics', 'inductive']
  for dataset in datasets:
    print(dataset)
    skip = False
    for method in methods:
      entropia = Entropia(dataset, method, datadir='/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_medium_ws2', namegetter=get_name)
      entropia.compute(verbose=True, skip=skip)
      if method == 'gcn_sound':
        if entropia.results['precision'] == 'nan' or entropia.results['precision'] is None:
          skip = True
      # entropia.export2()
      print(method, entropia.results)


if __name__ == '__main__':
  import argparse


  directory = {
    'simple': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions',
    'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws/predictions'
  }[method]
  mf_model = {'simple': 'gcn_ep144b', 'complex': 'gcn_ep100b'}[method]
  samples = {'simple': range(1970, 2627), 'complex': range(1488, 1985)}[method]

  datadir = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/'
  test_data()