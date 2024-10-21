import matplotlib.pyplot as plt
import numpy as np


def main():
  import pandas as pd
  import matplotlib.pyplot as plt

  plt.style.use('ggplot')

  values = [
    [60.6, 65.4, 32.2, 45.2],
    [5.4, 2.1, 24.9, 28.8],
    [32.2, 31.4, 37.2, 25.2],
    [1.8, 1.1, 5.7, 0.7]
  ]


  scales = [1.0, 0.7258146, 0.12374378, 0.274185]
  scales = [1.0, 0.8543434, 0.145656, 0.322738]
  values_scaled = [
    [v * i for i, v in zip(scales, values[0])],
    [v * i for i, v in zip(scales, values[1])],
    [v * i for i, v in zip(scales, values[2])],
    [v * i for i, v in zip(scales, values[3])]
  ]

  # pandas dataframe
  df = pd.DataFrame(data={'exactly specified': values_scaled[0], 'under- and overspecified': values_scaled[1], 'overspecified': values_scaled[2], 'underspecified': values_scaled[3]})
  df.index = ['Ground\ntruth', 'True\npositives', 'False\nnegatives', 'False\npositives']

  ax = df.plot(kind='bar', stacked=True, figsize=(5.5, 4), rot=0, xlabel='',
               ylabel='%', width=0.8, linewidth=2.5, color=['#8EBA42', '#348abd', '#e24a33', '#fbc15e'], alpha=0.75)
  for index, c in enumerate(ax.containers):
    # Optional: if the segment is small or 0, customize the labels
    labels = [f'{w:.1f}%' if v.get_height() > 1 else '' for v, w in zip(c, values[index])]
    heights = [v.get_height() for v in c]
    print(heights)
    # labels = values[index]
    print(labels)

    # remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=labels, label_type='center')

  plt.tight_layout()
  plt.savefig('/mnt/c/Users/s140511/tue/thesis/paper/figures/place_stats.pdf')
  plt.show()

if __name__ == '__main__':
  main()