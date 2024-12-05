from gnn_miner.data_handling.transition_invariants import compute_transition_invariants

from gnn_miner.data_handling.utility import transform_basis

def compute_t_components(net):
  transition_invariants = compute_transition_invariants(net)
  t_invariants = transform_basis(transition_invariants, style='uniform')
  """
  We perform the hint in 5.4.4 of https://pure.tue.nl/ws/portalfiles/portal/1596223/9715985.pdf
  :param p_invariants: Semi-positive basis we calculate previously
  :return: A list of S-Components. A s-component consists of a set which includes all related transitions a places
  """

  def compare_lists(list1, list2):
    """
    :param list1: a list
    :param list2: a list
    :return: a number how often a item from list1 appears in list2
    """
    counter = 0
    for el in list1:
      if el in list2:
        counter += 1
    return counter

  t_components = []
  transition_list = sorted(list(net.transitions), key=lambda x: str(x))
  for invariant in t_invariants:
    i = 0
    t_component = []
    for el in invariant:
      if el > 0:
        transition = transition_list[i]
        t_component.append(transition)
        for in_arc in transition.in_arcs:
          t_component.append(in_arc.source)
        for out_arc in transition.out_arcs:
          t_component.append(out_arc.target)
      i += 1

    if len(t_component) != 0:
      is_t_component = True
      for el in t_component:
        if el in net.places:
          transitions_before = [arc.source for arc in el.in_arcs]
          if compare_lists(t_component, transitions_before) != 1:
            is_t_component = False
            break
          transitions_after = [arc.target for arc in el.out_arcs]
          if compare_lists(t_component, transitions_after) != 1:
            is_t_component = False
            break
      if is_t_component:
        t_components.append(set(t_component))
  return t_components
