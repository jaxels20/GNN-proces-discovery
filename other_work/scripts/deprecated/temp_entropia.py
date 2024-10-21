import os
import shlex
import csv
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from threading import Timer
from pathlib import Path
import json
import tqdm

from project.data_handling.petrinet import PetrinetHandler
from project.evaluation.exporter import export_net

def get_precision_recall(command, verbose=False):
  if os.name == 'posix':
    command = shlex.split(command)

  results = {'precision': None, 'recall': None}
  keywords = {'precision': 'Precision: ', 'recall': 'Recall: '}

  process = Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
  for stdout_line in iter(process.stdout.readline, ""):
    if verbose:
      print(stdout_line, end='')
    for metric in ['precision', 'recall']:
      if stdout_line[:len(keywords[metric])] == keywords[metric]:
        results[metric] = float(stdout_line.split(' ')[1][:-2])

  process.stdout.close()
  process.wait()
  if verbose:
    print()
  return results

class Entropia:
  def __init__(self, log_filename, pnml_filename):
    self.basedir = '/mnt/c/Users/s140511/tue/thesis/codebase/jbpt-pm/entropia'
    self.log_filename = log_filename
    self.pnml_filename = pnml_filename
    self.results = {'precision': 'nan', 'recall': 'nan'}

  def prepare(self):
    filename = self.pnml_filename
    petri_net = PetrinetHandler()
    petri_net.importFromFile(filename)
    petri_net.removeStartAndEndTransitions()
    export_net(petri_net.mPetrinet, petri_net.mInitialMarking, f'{self.pnml_filename[:-5]}_temp.pnml', final_marking=petri_net.mFinalMarking)

  def cleanup(self):
    os.remove(f'{self.pnml_filename[:-5]}_temp.pnml')

  def compute(self, exact=True, verbose=False, skip=False):
    if skip:
      return
    try:
      self.prepare()
    except OSError:
      print('No file found')
      return
    if exact:
      command = f'java -jar {self.basedir}/jbpt-pm-entropia-1.6.jar -empr -rel={self.log_filename} -ret={self.pnml_filename[:-5]}_temp.pnml'
    else:
      command = f'java -jar {self.basedir}/jbpt-pm-entropia-1.6.jar -pmpr -rel={self.log_filename} -ret={self.pnml_filename[:-5]}_temp.pnml'
    print(command)
    if verbose:
      print()
      print()
    self.results = get_precision_recall(command, verbose=verbose)
    if verbose:
      print()
      print()
    self.cleanup()

  # def export(self):
  #   filename = f'{self.datadir}/{self.dataset}/results/data_{self.method}_cc_temp.txt'
  #   with open(filename, 'r') as file:
  #     data = json.load(file)
  #     data['full'][f'{"entropia_partial_precision":<32}'] = str(self.results['precision'])
  #     data['full'][f'{"entropia_partial_recall":<32}'] = str(self.results['recall'])
  #   with open(f'{self.datadir}/{self.dataset}/results/data_{self.method}_cc_temp2.txt', 'w') as f:
  #     json.dump(data, f, indent=2)
  #
  # def export2(self):
  #   basedir = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_medium_ws2'
  #   filename = f'{basedir}/logs/predictions/new_conformance.csv'
  #   fields = [f'{self.dataset}/{self.method}', self.results['precision'], self.results['recall']]
  #   with open(filename, 'a') as f:
  #     writer = csv.writer(f)
  #     writer.writerow(fields)

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


# def get_name(dataset, method):
#   basedir = f'/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_medium_ws2'
#   if method == 'gcn_sound':
#     return f'{basedir}/logs_compressed/predictions/{dataset}_gcn_sound.pnml'
#   elif method == 'groundtruth':
#     return f'{basedir}/petrinets/{dataset}.pnml'
#   else:
#     return f'{basedir}/logs/predictions/{dataset}_{method}.pnml'

def get_soundness(pnml_filename):
  pnet = PetrinetHandler()
  pnet.importFromFile(pnml_filename)
  soundness = pnet.get_pm4py_soundness(timeout=60)
  if soundness[0]:
    easy_soundness = True
  else:
    easy_soundness = pnet.get_easy_soundness(timeout=15)
  return soundness[0], easy_soundness


def eval_data():
  # datadir = '/mnt/c/Users/s140511/tue/thesis/thesis_data/evaluation_data/'
  datasets = ['BPI_2012_A', 'BPI_2012_O', 'BPI_2017_A', 'BPI_2017_O', 'BPI_2020_Domestic_declarations',
              'BPI_2020_International_declarations', 'BPI_2020_Permit_log', 'BPI_2020_Prepaid_travel_cost',
              'BPI_2020_Request_for_payment', 'road_traffic_fine', 'sepsis']
  datasets = ['BPI_2020_Prepaid_travel_cost', 'BPI_2020_Request_for_payment', 'road_traffic_fine', 'sepsis']
  methods = ['split_reduced', 'gcn', 'inductive_reduced', 'heuristics_reduced', 'ilp_reduced']
  for dataset in tqdm.tqdm(datasets):
    print(dataset)
    for method in methods:
      if method == 'gcn':
        method = find_gcn_model(dataset)

      entropia = Entropia(dataset, method)
      entropia.compute(exact=False, verbose=True)
      entropia.export()
      print(method, entropia.results)

def test_data(samples, methods, directory):
  samples = [f'{i:04d}' for i in samples]
  for sample in tqdm.tqdm(samples):
    skip = False
    log_filename = f"{'/'.join(directory.split('/')[:-1])}/logs/{sample}.xes"
    for method in methods:
      if method == 'groundtruth':
        pnml_filename = f"{'/'.join(directory.split('/')[:-1])}/petrinets/{sample}.pnml"
      else:
        pnml_filename = f'{directory}/{sample}_{method}.pnml'

      if not os.path.exists(pnml_filename):
        print(f'{pnml_filename} file not exists.')
        continue

      cc_filename = f'{directory}/{sample}_{method}_cca.json'
      if not os.path.exists(cc_filename):
        print(f'{cc_filename} file not exists.')
        continue

      print(cc_filename)
      with open(cc_filename, 'r') as f:
        cc = json.load(f)
      if 'easy_soundness' not in cc.keys():
        cc['sound'], cc['easy_soundness'] = get_soundness(pnml_filename)
      sound, easy_sound = cc['sound'], cc['easy_soundness']

      print('SOUND', sound, easy_sound)
      if not easy_sound:
        skip = True

      entropia = Entropia(log_filename, pnml_filename)
      entropia.compute(verbose=True, skip=skip)
      if 'gcn' in method:
        if entropia.results['precision'] == 'nan' or entropia.results['precision'] is None:
          skip = True

      print(method, entropia.results)
      if 'full' in cc:
        cc['full']['entropia_recall'] = entropia.results['recall']
        cc['full']['entropia_precision'] = entropia.results['precision']
      elif 'FULL' in cc:
        cc['FULL']['entropia_recall'] = entropia.results['recall']
        cc['FULL']['entropia_precision'] = entropia.results['precision']
      else:
        cc['entropia_recall'] = entropia.results['recall']
        cc['entropia_precision'] = entropia.results['precision']

      export = True
      if export:
        with open(f'{cc_filename[:-5]}.json', 'w') as f:
          json.dump(cc, f, sort_keys=True, indent=2)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-ds', '--dataset', help='dataset', type=str)
  args = parser.parse_args()

  method = args.dataset
  directory = {
    'simple':  '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/predictions',
    'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws/predictions',
    'small': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees/predictions'
  }[method]
  mf_model = {'simple': 'gcn_ep144b', 'complex': 'gcn_ep100b', 'small': 'gcn33'}[method]
  samples = {'simple': range(1970, 2627), 'complex': range(1488, 1985), 'small': range(0, 734)}[method]
  samples = {'simple': range(2192, 2627), 'complex': range(1560, 1985), 'small': range(511, 734)}[method]

  methods = ['groundtruth', 'split', 'heuristics', 'inductive']
  methods.insert(1, mf_model)
  methods = [mf_model]

  # test_data([1970], methods, directory)
  test_data(samples, methods, directory)




  # results = get_precision_recall(f'java -jar {basedir}/jbpt-pm-entropia-1.6.jar -empr -rel={datadir}/{dataset}/data.xes -ret={datadir}/{dataset}/results/data_{method}.pnml')
  # print(results)

  # results = {'precision': None, 'recall': None}
  # keywords = {'precision': 'Precision: ', 'recall': 'Recall: '}
  # for line in exec(f'java -jar {dir}/jbpt-pm-entropia-1.6.jar -empr -rel={dir}/data/data.xes -ret={dir}/data/inductive.pnml'):
  #   for metric in ['precision', 'recall']:
  #     if line[:len(keywords[metric])] == keywords[metric]:
  #       results[metric] = float(line.split(' ')[1][:-2])
  # print(results)
