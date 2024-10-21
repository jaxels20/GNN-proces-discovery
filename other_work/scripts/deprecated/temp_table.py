from rich.pretty import pprint
import numpy as np

dataset_names = {
  'road_traffic_fine': ["Road", "Traffic", "Fine"],
  'sepsis': ["Sepsis"],
  'BPI_2012_A': ["'12", "A\_", "prefixed"],
  'BPI_2012_O': ["'12", "O\_", "prefixed"],
  'BPI_2017_A': ["'17", "A\_", "prefixed"],
  'BPI_2017_O': ["'17", "O\_", "prefixed"],
  'BPI_2020_Domestic_declarations': ["'20", "Domestic", "Declarations"],
  'BPI_2020_International_declarations': ["'20", "International", "Declarations"],
  'BPI_2020_Permit_log': ["'20", "Permit Log"],
  'BPI_2020_Prepaid_travel_cost': ["'20", "Prepaid", "Travel Cost"],
  'BPI_2020_Request_for_payment': ["'20", "Request for", "Payment"],
  'Average': ['Average'],
}

methods = {
  'gcn': 'Ours',
  'split_reduced': 'SM',
  'heuristics_reduced': 'HM',
  'inductive_reduced': 'IM'
}

methods_test = {
  'Ground truth': 'GT',
  'Our approach': 'Ours',
  'Split Miner': 'SM',
  'Heuristics Miner': 'HM',
  'Inductive Miner': 'IM'
}

def get_values(data, dataset, method):
  if dataset == 'Average':
    m = np.matrix(data[dataset][method])
    values = []
    for col in range(m.shape[1]):
      non_nuls = [v for v in m[:,col] if v > 0]
      if len(non_nuls) == 0:
        values.append('n/a')
      else:
        values.append(f'{np.mean(non_nuls):.2f}')
    return values

  if method in data[dataset]:
    values = [f'{v:.2f}' if v > 0.0 else 'n/a' for v in data[dataset][method]]
  else:
    values = ['n/a' for _ in range(len(data[dataset]['split_reduced']))]
  return values

def get_values_test(data, dataset, method):
  if dataset == 'Average':

    m = np.matrix(data[dataset][method])
    values = []
    for col in range(m.shape[1]):
      non_nuls = [v for v in m[:,col] if v > 0]
      if len(non_nuls) == 0:
        values.append('n/a')
      else:
        values.append(f'{np.mean(non_nuls):.2f}')
    return values

  if method in data[dataset]:
    values = [f'{v0:.2f} ({v1:.2f})' if v0 > 0.0 else 'n/a' for (v0, v1) in data[dataset][method]]
  else:
    values = ['n/a' for _ in range(len(data[dataset]['split_reduced']))]
  return values

def table_test(data):
  pprint(data)
  datasets = {'test': data}
  method_values = {}
  for i, (method, method_label) in enumerate(methods_test.items()):
    method_values[method] = get_values_test(datasets, 'test', method)
    print(f"{method_label} & {' & '.join(method_values[method])} \\\\")
    # pprint(method_values)
  print(fdsa)

def table(datasets):
  pprint(datasets)
  for ds, ds_name in dataset_names.items():
    method_values = {}
    for i, (method, method_label) in enumerate(methods.items()):
      method_values[method] = get_values(datasets, ds, method)
    # Mark best values
    for i in range(len(method_values['gcn'])):
      vals = [0 if vv[i] == 'n/a' else float(vv[i]) for vv in method_values.values()]
      maximal = max(vals)
      for m in method_values.keys():
        if method_values[m][i] != 'n/a' and float(method_values[m][i]) == maximal:
          method_values[m][i] = '\\bf{' + method_values[m][i] + '}'

    for i, (method, method_label) in enumerate(methods.items()):
      if method == 'gcn':
        print(ds)
        print(f"& {method_label} (c) & {' & '.join(method_values[method])} \\\\")
      # ds_prefix = '' if i >= len(ds_name) else ds_name[i]
      # print(f"{ds_prefix} & {method_label} & {' & '.join(method_values[method])} \\\\")
    # print('\\hline')

  print(datasets.keys())
