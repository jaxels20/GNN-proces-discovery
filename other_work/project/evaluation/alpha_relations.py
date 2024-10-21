import numpy as np
from collections import defaultdict
from pprint import pprint


class AlphaRelations:
  def __init__(self, variants, transitions_names):
    self.variants = variants
    self.__add_start_end()
    self.transition_mapping = self.__get_transition_mapping(list(transitions_names))
    self.directly_follows_relations = self.__get_directly_follows_relations()
    self.directly_follows_relations_dict = self.__get_directly_follows_relations_dict(self.directly_follows_relations)
    self.causal_relations = self.__get_causal_relations()
    self.parallel_relations = self.__get_parallel_relations()

  def get_matrix(self, names=False):
    from prettytable import PrettyTable
    legend = list(self.transition_mapping.values()) if names else list(self.transition_mapping.keys())

    matrix = [[''] + legend]
    for t in legend:
      matrix.append([t] + ['#'] * len(legend))

    for t1, t2 in self.directly_follows_relations:
      matrix[t1 + 1][t2 + 1] = '-'

    for t1, t2 in self.causal_relations:
      matrix[t1 + 1][t2 + 1] = '>'

    for t1, t2 in self.parallel_relations:
      matrix[t1 + 1][t2 + 1] = '||'

    table = PrettyTable(matrix[0])
    for row in matrix[1:]:
      table.add_row(row)

    return matrix, table

  def __add_start_end(self):
    newvariants = []
    for variant in self.variants:
      newvariants.append([0] + [t + 2 for t in variant] + [1])
    self.variants = newvariants

  def __get_transition_mapping(self, transitions_names):
    return {**{0: '>', 1: '|'}, **dict(zip(range(2, len(transitions_names) + 2), transitions_names))}

  def __get_directly_follows_relations(self):
    return set([(t1, t2) for variant in self.variants for t1, t2 in zip(variant[:-1], variant[1:])])

  def __get_directly_follows_relations_dict(self, directly_follows_relations):
    d = defaultdict(set)
    for relation_from, relation_to in directly_follows_relations:
      d[relation_from].add(relation_to)
    return d

  def __get_causal_relations(self):
    return set([(t1, t2) for (t1, t2) in self.directly_follows_relations if (t2, t1) not in self.directly_follows_relations])

  def __get_parallel_relations(self):
    return set([(t1, t2) for (t1, t2) in self.directly_follows_relations if (t2, t1) in self.directly_follows_relations])
