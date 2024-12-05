import numpy as np
from colorama import Style, Fore, Back


def softmax(xx):
  return np.exp(xx) / np.sum(np.exp(xx), axis=0)


def print_variants(variants, labels, counts=None):
  if counts is None:
    counts = [''] * len(variants)

  colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX,
            Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTWHITE_EX]
  colors = [Back.RED, Fore.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE, Back.LIGHTRED_EX,
            Back.LIGHTGREEN_EX, Back.LIGHTYELLOW_EX, Back.LIGHTBLUE_EX, Back.LIGHTMAGENTA_EX, Back.LIGHTCYAN_EX, Back.LIGHTWHITE_EX]

  for variant, count in zip(variants, counts):
    variant_string = ''
    for event in variant:
      variant_string += f'{colors[labels.index(event) % len(colors)]}{event}{Style.RESET_ALL},'
    print(f'{variant_string} {count}')
