from project.data_handling.petrinet import PetrinetHandler

from pm4py import write_xes, write_pnml

import os
import shlex
import csv
from subprocess import Popen, PIPE, STDOUT
import json
from threading import Timer


def get_precision_recall(command, timeout=-1, verbose=False):
  if os.name == 'posix':
    command = shlex.split(command)

  results = {'precision': None, 'recall': None}
  keywords = {'precision': 'Precision: ', 'recall': 'Recall: '}

  process = Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
  if timeout <= 0:
    for stdout_line in iter(process.stdout.readline, ""):
      if verbose:
        print(stdout_line, end='')
      for metric in ['precision', 'recall']:
        if stdout_line[:len(keywords[metric])] == keywords[metric]:
          results[metric] = float(stdout_line.split(' ')[1][:-2])
    process.stdout.close()
    process.wait()
  else:
    timer = Timer(timeout, process.kill)
    try:
      timer.start()
      for stdout_line in iter(process.stdout.readline, ""):
        if verbose or True:
          print(stdout_line, end='')
        for metric in ['precision', 'recall']:
          if stdout_line[:len(keywords[metric])] == keywords[metric]:
            results[metric] = float(stdout_line.split(' ')[1][:-2])
      process.stdout.close()
      process.wait()
      timer.cancel()
    finally:
      if not timer.is_alive():
        print(f'Entropia time out ({timeout}s).')

  if verbose:
    print()
  return results

class Entropia:
  def __init__(self, log, petri_net, m_i, m_f, entropia_base_dir, temp_filename=''):
    self.log = log
    self.petri_net = petri_net
    self.m_i = m_i
    self.m_f = m_f
    self.entropia_base_dir = entropia_base_dir # '/mnt/c/Users/s140511/tue/thesis/codebase/jbpt-pm/entropia'
    self.entropia_jar_filename = f'{entropia_base_dir}/jbpt-pm-entropia-1.6.jar'
    self.results = {'precision': None, 'recall': None}
    self.temp_filename = temp_filename

  def prepare(self):
    self.petri_net_filename = f'{self.entropia_base_dir}/data/temp{self.temp_filename}.pnml'
    write_pnml(self.petri_net, self.m_i, self.m_f, self.petri_net_filename)
    self.log_filename = f'{self.entropia_base_dir}/data/temp{self.temp_filename}.xes'
    write_xes(self.log, self.log_filename)
    print('entropia', self.petri_net_filename, self.log_filename)

  def cleanup(self):
    os.remove(self.petri_net_filename)
    os.remove(self.log_filename)

  def compute(self, exact=True, timeout=-1, verbose=False, skip=False):
    if skip:
      return
    try:
      self.prepare()
    except OSError:
      print('No file found')
      return
    if exact:
      command = f'java -jar {self.entropia_jar_filename} -empr -rel={self.log_filename} -ret={self.petri_net_filename}'
    else:
      command = f'java -jar {self.entropia_jar_filename} -pmpr -rel={self.log_filename} -ret={self.petri_net_filename}'

    if verbose:
      print()
      print()
    self.results = get_precision_recall(command, timeout=timeout, verbose=verbose)
    if verbose:
      print()
      print()
    self.cleanup()
    return self.results

  # # TODO fix hardcoded paths.
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
