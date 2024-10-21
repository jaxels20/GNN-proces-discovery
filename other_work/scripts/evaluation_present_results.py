import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

'''
Figure 10(a)
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fitness_entropy_partial precision_entropy_partial fitness_alignments precision_alignments
       -a gcn_medium_all split_reduced heuristics_reduced inductive_reduced

Figure 10(b)
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fscore_entropy_partial fscore_alignments simplicity
       -a gcn_medium_all split_reduced heuristics_reduced inductive_reduced

Table 2
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fitness_entropy_partial precision_entropy_partial fscore_entropy_partial fitness_alignments precision_alignments fscore_alignments simplicity
       -a gcn_medium_all split_reduced heuristics_reduced inductive_reduced
       
Figure 15(a)
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fitness_entropy_partial precision_entropy_partial fitness_alignments precision_alignments
       -a gcn_medium_all gcn_simple_all gcn_complex_all

Figure 15(b)
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fscore_entropy_partial fscore_alignments simplicity
       -a gcn_medium_all gcn_simple_all gcn_complex_all

Table 6
python3 scripts/evaluation_present_results.py -d /mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data 
       -m fitness_entropy_partial precision_entropy_partial fscore_entropy_partial fitness_alignments precision_alignments fscore_alignments simplicity
       -a gcn_medium_all gcn_simple_all gcn_complex_all
'''

plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', help='data directory', type=str)
parser.add_argument('-m', '--metrics', help='Metrics', nargs='+', action='append')
parser.add_argument('-a', '--approaches', help='Approaches', nargs='+', action='append')
parser.add_argument('-e', '--export', help='export', action='store_true')
parser.add_argument('-pt', '--print_table', action='store_true')
args = parser.parse_args()

base_dir = args.data_directory    # '/mnt/c/Users/s140511/tue/thesis/APDGnn_data/evaluation_data'
metrics_to_plot = args.metrics[0] # ['fscore_alignments', 'fscore_entropy_partial', 'simplicity']
approaches = args.approaches[0]   # ['gcn_medium_all', 'split_reduced', 'heuristics_reduced', 'inductive_reduced']

datasets = {
  'road_traffic_fine': "Road Traffic\nFine", 'sepsis': "Sepsis",
  'BPI_2012_A': "'12\nA_ prefixed", 'BPI_2012_O': "'12\nO_ prefixed", 'BPI_2017_A': "'17\nA_ prefixed",
  'BPI_2017_O':"'17\nO_ prefixed", 'BPI_2020_Domestic_declarations': "'20\nDomestic\nDeclarations", 'BPI_2020_International_declarations': "'20\nInternational\nDeclarations",
  'BPI_2020_Permit_log': "'20\nPermit Log\n", 'BPI_2020_Prepaid_travel_cost': "'20\nPrepaid\nTravel Cost", 'BPI_2020_Request_for_payment': "'20\nRequest for\nPayment",
  'Average': 'Average\n'
}

methods = {
  'gcn_medium_all': 'Our approach', 'gcn_simple_all': 'Our approach (simple)', 'gcn_complex_all': 'Our approach (complex)',
  'ilp_reduced': 'ILP Miner', 'split': 'Split Miner', 'split_reduced': 'Split Miner', 'heuristics_reduced': 'Heuristics Miner', 'inductive_reduced': 'Inductive Miner', 'alpha': 'Alpha Miner',
}

methods_short = {
  'gcn_medium_all': 'Ours', 'gcn_simple_all': 'low', 'gcn_complex_all': 'high',
  'split_reduced': 'SM', 'heuristics_reduced': 'HM', 'inductive_reduced': 'IM'
}

metrics_labels = {
  'fscore': 'F-score_', 'fitness': 'Fitness_', 'precision': 'Precision_', 'simplicity': 'Simplicity', 'generalization': 'Generalization', 'metricsAverageWeight': 'metricsAverageWeight',
  'fscore_alignments': 'F-score\n(alignment)', 'fitness_alignments': 'Fitness\n(alignment)', 'precision_alignments': 'Precision\n(alignment)',
  'fscore_entropy_partial': 'F-score\n(entropy)', 'fitness_entropy_partial': 'Fitness\n(entropy)', 'precision_entropy_partial': 'Precision\n(entropy)',
'fscore_entropy_exact': 'F-score\n(entropy)', 'fitness_entropy_exact': 'Fitness\n(entropy)', 'precision_entropy_exact': 'Precision\n(entropy)',
}

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
approach_colors = {
  'Our approach (intermediate)': colors[0], 'Our approach (simple)': colors[1], 'Our approach (complex)': colors[5], 'Our approach': colors[0],
  'Split Miner': colors[2], 'Heuristics Miner': colors[3], 'Inductive Miner': colors[4],
}


# Retrieve conformance results from files.
data = pd.DataFrame()
for dataset_i, dataset_name in enumerate(datasets.keys()):
  if dataset_name == 'Average':
    continue
  print(dataset_name)
  for approach_i, approach in enumerate(approaches):
    conformance_filename = f'{base_dir}/{dataset_name}/results/data_{approach}_cc.json'
    if os.path.exists(conformance_filename):
      with open(conformance_filename, 'r') as conformance_file:
        conformance = json.load(conformance_file)
      if isinstance(conformance, list):
        sort_on_fscore = 'fscore_alignments' # or entropy_partial # TODO in case of entropy_partial, take fscore_alignment when entropy_partial is none.
        sort_lambda = lambda c: int(c['sound']) + int(c['easy_soundness']) + (c['L']['fscore'] if c['L'][sort_on_fscore] is None else c['L'][sort_on_fscore])
        conformance = list(sorted(conformance, key=sort_lambda, reverse=True))[0]
    else:
      print(f'Conformance file: {conformance_filename} does not exist.')
      conformance = {'L': {}}
    data = data.append({'dataset': dataset_name, 'dataset_i': dataset_i, 'approach': approach, 'approach_i': approach_i,
                        **conformance['L']}, ignore_index=True)

data = data.fillna(0)
# Clip values between 0 and 1.
data.loc[:, data.select_dtypes(include=[np.number]).columns].clip(0, 1, inplace=True)
print(data[['dataset', 'approach', *metrics_to_plot]])


# Set average, excluding zeroes!!
data = data.replace(0, np.NaN)
for approach_i, approach in enumerate(approaches):
  mean_scores = dict(data[data['approach'] == approach][metrics_to_plot].mean())
  data = data.append({'dataset': 'Average', 'dataset_i': list(datasets.keys()).index('Average'), 'approach': approach,
                      'approach_i': approach_i, **mean_scores}, ignore_index=True)
data = data.fillna(0)

def plot():
  rows, cols = len(metrics_to_plot), 1
  fig, ax = plt.subplots(rows, cols, figsize=(15, len(metrics_to_plot)*1.2 + 0.7))

  width = 0.2  # the width of the bars
  offsets = {4: [-1.5*width, -0.5*width, 0.5*width, 1.5*width],
             3: [-1*width, 0, 1*width]}

  x = np.arange(len(datasets.keys()))
  for ax_i, metric in enumerate(metrics_to_plot):
    for index, approach in enumerate(approaches):
      approach_label = methods[approach]
      if 'gcn_medium' in approach and len([x for x in approaches if 'gcn' in x]) > 1:
        approach_label = f'{approach_label} (intermediate)'
      scores = list(data[data['approach'] == approach][metric])
      ax[ax_i].bar(x + offsets[len(approaches)][index], scores, width, label=approach_label if ax_i == 0 else None, alpha=.7,
                  edgecolor='k', color=approach_colors[approach_label])

  for metric_i, metric in enumerate(metrics_to_plot):
    ax[metric_i].set_xticks(x)
    ax[metric_i].set_ylabel(metrics_labels[metric], fontsize=13)
    if metric_i == len(metrics_to_plot) - 1: # Bottom sub plot
      ax[metric_i].set_xticklabels(list(datasets.values()), fontsize=13)
    else:
      ax[metric_i].set_xticklabels([''] * len(datasets), fontsize=13)

    ax[metric_i].set_xlim(-0.50, 11.5)
    ax[metric_i].set_ylim(-0.01, 1.01)
    # ax[metric_i].set_ylim( 0.25, 1.01)

  legend_h = {3: 1.017, 4: 0.995}.get(len(metrics_to_plot), 1)
  fig.legend(loc='upper center', bbox_to_anchor=(0.5, legend_h), ncol=len(approaches), fancybox=False, fontsize=13)
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.1, top=0.92)

  if args.export:
    filename = f'scripts/figures/evaluation_bar_{"_".join(metrics_to_plot)}_{"_".join([methods_short[m] for m in approaches])}.pdf'
    print(f'Exporting {filename}')
    fig.savefig(filename)
  plt.show()


def print_table():
  multiline_datasets = {
    'road_traffic_fine': ['Road', 'Traffic', 'Fine'], 'sepsis': ['Sepsis'],
    'BPI_2012_A': ["'12","A\_","prefixed"], 'BPI_2012_O': ["'12","O\_","prefixed"], 'BPI_2017_A': ["'17","A\_","prefixed"],
    'BPI_2017_O': ["'17","O\_","prefixed"], 'BPI_2020_Domestic_declarations': ["'20","Domestic","Declarations"], 'BPI_2020_International_declarations': ["'20","International","Declarations"],
    'BPI_2020_Permit_log': ["'20","Permit Log"], 'BPI_2020_Prepaid_travel_cost': ["'20","Prepaid","Travel Cost"], 'BPI_2020_Request_for_payment': ["'20","Request for","Payment"],
    'Average': ['Average']
  }
  table_data = data.sort_values(by=['dataset_i', 'approach_i'])[['dataset', 'approach', *metrics_to_plot]]

  dataset_counter = 0
  current_dataset = None
  lines = []
  for i, row in table_data.iterrows():
    dataset = row['dataset']
    if current_dataset != dataset:
      current_dataset = dataset
      dataset_counter = 0
      lines.append('\\hline')
      print(lines[-1])
      dataset_prefix = multiline_datasets[dataset][0]
    else:
      dataset_counter += 1
      dataset_prefix = multiline_datasets[dataset][dataset_counter] if len(multiline_datasets[dataset]) > dataset_counter else ''

    is_best = lambda v, m: v > 0 and table_data[table_data['dataset'] == dataset][m].max() == v
    scores = [('\\bf{' if is_best(row[m], m) else '   {') +  (f'{row[m]:.2f}' if row[m] > 0 else ' n/a') + '}' for m in metrics_to_plot]

    if 'gcn_medium' in row['approach'] and len([a for a in approaches if 'gcn' in a]) > 1:
      short_method_label = 'medium'
    else:
      short_method_label = methods_short[row['approach']]
    lines.append(f"{dataset_prefix:<13} & {short_method_label:<6} & {' & '.join(scores)} \\\\")
    print(lines[-1])

  if args.export:
    filename = f'scripts/figures/evaluation_table_{"_".join([methods_short[m] for m in approaches])}.txt'
    print(f'Exporting {filename}')
    with open(filename, 'w') as file:
      file.write('\n'.join(lines))



if args.print_table:
  print_table()
else:
  plot()
