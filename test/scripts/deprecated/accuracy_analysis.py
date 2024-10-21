import matplotlib.pyplot as plt
import json
import numpy as np
import argparse


plt.rcParams.update({'text.usetex': True})

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--new', action='store_true')
args = parser.parse_args()

if args.new:
  modelname = 'model_candidates'
else:
  modelname = 'model_candidates_frequency_025'

modelname = 'model_candidates_frequency'

train_filename = f'/mnt/c/Users/s140511/tue/thesis/git/project/ml_models/checkpoints/{modelname}_train_stats.json'
test_filename  = f'/mnt/c/Users/s140511/tue/thesis/git/project/ml_models/checkpoints/{modelname}_test_stats.json'

with open(train_filename, 'r') as file:
  train_statistics = json.load(file)
with open(test_filename, 'r') as file:
  test_statistics = json.load(file)

# print(train_statistics)

all_statistics = {'train': train_statistics, 'test': test_statistics}
results = {'train': {'tps': [], 'fns': [], 'losses': []},
           'test':  {'tps': [], 'fns': [], 'losses': []}}
for key, statistics in all_statistics.items():
  print(key)
  for epoch_stats in statistics:
    # tps_epoch = [stats['tp'] for stats in epoch_stats]
    # fns_epoch = [stats['fn'] for stats in epoch_stats]
    losses_epoch = [stats['loss'] if 'loss' in stats else 0 for stats in epoch_stats]
    # results[key]['tps'].append(tps_epoch)
    # results[key]['fns'].append(fns_epoch)
    results[key]['losses'].append(losses_epoch)

plt.style.use('ggplot')
# plt.style.use('seaborn-pastel')

# fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for row, trainTest in enumerate(['train', 'test']):
  for col, (statKey, title, ylim, ylabel) in enumerate(zip(
      ['losses', 'tps', 'fns'], ['Loss', '\% True positives', '\% False positives'],
      # [[0, 25], [0.85, 1], [0, 0.25]],
      # [[0, 50], [0.5, 1], [0, 1.5]],
      [[0, 40], [0.7, 1], [0, 3]],
      # [[0, 50], [0.3, 1], [0, 3]],
      [r'$loss = -\sum_p \log p$', '$$\\frac{\\# true\_positives}{\\#places}$$', '$$\\frac{\\# false\_positives}{\\#places}$$'])):
    # boxes = [a for i, a in enumerate(results[trainTest][statKey]) if i % 5 == 0 or i == len(results[trainTest][statKey]) - 1]
    # axs[row, col].boxplot(boxes, positions=[i * 5 + int(i != len(boxes) - 1) for i in range(len(boxes))], widths=3, manage_ticks=False)
    axs[col].set_xlabel('epoch')
    axs[col].set_ylabel(ylabel)
    # if trainTest == 'test':
    #   axs[row, col].plot([0], [0])
    axs[col].plot(range(1, len(results[trainTest][statKey]) + 1), [np.mean(data_epoch) for data_epoch in results[trainTest][statKey]], linewidth=2, label=f'{trainTest}')
    # axs[row, col].legend()
    axs[col].set_ylim(ylim[0], ylim[1])
    # axs[row, col].set_title(f'{title} ({trainTest})')
    axs[col].set_title(f'{title}')

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.tight_layout()
# fig.savefig(f'/home/dominique/TUe/thesis/report/eval_figures/training_curves_scratch.pdf')

# fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)
plt.show()
