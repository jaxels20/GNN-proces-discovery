from prettytable import PrettyTable


class PlaceEvaluation:
  def __init__(self, input_transitions, output_transitions, alpha_relations=None):
    self.input_transitions = input_transitions
    self.output_transitions = output_transitions
    self.alpha_relations = alpha_relations
    self.sub_alpha_relations = self.get_alpha_relations(alpha_relations)
    self.specified = set()
    self.under_specified = set()
    self.over_specified = set()
    self.wrong_parallel_relations = set()

  def get_alpha_relations(self, alpha_relations):
    # TODO also get the relations between the output transitions
    matrix, table = alpha_relations.get_matrix(names=True)
    row_indices = [index for index, row in enumerate(matrix) if row[0] in self.input_transitions]
    col_indices = [index for index, name in enumerate(matrix[0]) if name in self.output_transitions]
    subtable = [row for row_index, row in enumerate(matrix) if row_index == 0 or row_index in row_indices]
    subsubtable = PrettyTable([value for col_index, value in enumerate(subtable[0]) if col_index == 0 or col_index in col_indices])
    for row in subtable[1:]:
      subsubtable.add_row([value for col_index, value in enumerate(row) if col_index == 0 or col_index in col_indices])

    self.directly_follows = set(
      [(self.alpha_relations.transition_mapping[relation[0]], self.alpha_relations.transition_mapping[relation[1]]) for
       relation in self.alpha_relations.directly_follows_relations if
       self.alpha_relations.transition_mapping[relation[0]] in self.input_transitions or
       self.alpha_relations.transition_mapping[relation[1]] in self.output_transitions])

    self.causal_relations = set(
      [(self.alpha_relations.transition_mapping[relation[0]], self.alpha_relations.transition_mapping[relation[1]]) for
       relation in self.alpha_relations.causal_relations if
       self.alpha_relations.transition_mapping[relation[0]] in self.input_transitions or
       self.alpha_relations.transition_mapping[relation[1]] in self.output_transitions])

    self.parallel_relations = set(
      [(self.alpha_relations.transition_mapping[relation[0]], self.alpha_relations.transition_mapping[relation[1]]) for
       relation in self.alpha_relations.parallel_relations if
       self.alpha_relations.transition_mapping[relation[0]] in self.input_transitions or
       self.alpha_relations.transition_mapping[relation[1]] in self.output_transitions])

    return subsubtable

  def analyze_information(self):
    for input_transition in self.input_transitions:
      for output_transition in self.output_transitions:
        causal_relation = (input_transition,  output_transition)
        if causal_relation in self.causal_relations:
          self.specified.add(causal_relation)
        else:
          self.under_specified.add(causal_relation)

    for parallel_relation in self.parallel_relations:
      if (parallel_relation[0] in self.input_transitions and parallel_relation[1] in self.input_transitions) or \
         (parallel_relation[0] in self.output_transitions and parallel_relation[1] in self.output_transitions):
        self.wrong_parallel_relations.add(parallel_relation)

    for causal_relation in self.causal_relations:
      if causal_relation not in self.specified:
        self.over_specified.add(causal_relation)

    for relation in [x for x in self.over_specified]:
      for relation2 in self.specified:
        if (relation[0], relation2[0]) in self.parallel_relations or (relation[1], relation2[1]) in self.parallel_relations or \
           (relation2[0], relation[0]) in self.parallel_relations or (relation2[1], relation[1]) in self.parallel_relations:
          self.over_specified.discard(relation)
    self.over_specified.update(self.wrong_parallel_relations.copy())

  def check_contained(self, other):
    return (self.input_transitions.issubset(other.input_transitions) and other.output_transitions == self.output_transitions) or \
           (self.output_transitions.issubset(other.output_transitions) and other.input_transitions == self.input_transitions) or \
           (self.input_transitions.issubset(other.input_transitions) and self.output_transitions.issubset(other.output_transitions))
