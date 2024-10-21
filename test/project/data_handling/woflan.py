import networkx as nx
from pm4py.evaluation.soundness.woflan.algorithm import short_circuit_petri_net
from pm4py.objects.petri_net.obj import PetriNet

# Importing for place invariants related stuff (s-components, uniform and weighted place invariants)
from pm4py.evaluation.soundness.woflan.place_invariants.place_invariants import compute_place_invariants
from pm4py.evaluation.soundness.woflan.place_invariants.utility import transform_basis
from pm4py.evaluation.soundness.woflan.place_invariants.s_component import compute_s_components
from pm4py.evaluation.soundness.woflan.place_invariants.s_component import compute_uncovered_places_in_component
from pm4py.evaluation.soundness.woflan.place_invariants.utility import \
    compute_uncovered_places as compute_uncovered_place_in_invariants

def step_1_(woflan_object, return_asap_when_unsound=False):
  """
  In the first step, we check if the input is given correct. We check if net is an PM4Py Petri Net representation
  and if the exist a correct entry for the initial and final marking.
  :param woflan_object: Object that contains all necessary information
  :return: Proceed with step 2 if ok; else False
  """

  def check_if_marking_in_net(marking, net):
    """
    Checks if the marked place exists in the Petri Net and if there is only one i_m and f_m
    :param marking: Marking of Petri Net
    :param net: PM4Py representation of Petri Net
    :return: Boolean. True if marking can exists; False if not.
    """
    for place in marking:
      if place in net.places:
        return True
    return False

  if isinstance(woflan_object.get_net(), PetriNet):
    if len(woflan_object.get_initial_marking()) != 1 or len(woflan_object.get_final_marking()) != 1:
      if woflan_object.print_diagnostics:
        print('There is more than one initial or final marking.')
      return False
    if check_if_marking_in_net(woflan_object.get_initial_marking(), woflan_object.get_net()):
      if check_if_marking_in_net(woflan_object.get_final_marking(), woflan_object.get_net()):
        if woflan_object.print_diagnostics:
          print("Input is ok.")
        return True #step_2_(woflan_object, return_asap_when_unsound=return_asap_when_unsound)
  if woflan_object.print_diagnostics:
    print('The Petri Net is not PM4Py Petri Net represenatation.')
  return False


def step_2_(woflan_object, return_asap_when_unsound=False, already_short_circuited=True):
  """
  This method checks if a given Petri net is a workflow net. First, the Petri Net gets short-circuited
  (connect start and end place with a tau-transition. Second, the Petri Net gets converted into a networkx graph.
  Finally, it is tested if the resulting graph is a strongly connected component.
  :param woflan_object: Woflan objet containing all information
  :return: Bool=True if net is a WF-Net
  """

  def transform_petri_net_into_regular_graph(still_need_to_discover):
    """
    Ths method transforms a list of places and transitions into a networkx graph
    :param still_need_to_discover: set of places and transition that are not fully added to graph
    :return:
    """
    G = nx.DiGraph()
    while len(still_need_to_discover) > 0:
      element = still_need_to_discover.pop()
      G.add_node(element.name)
      for in_arc in element.in_arcs:
        G.add_node(in_arc.source.name)
        G.add_edge(in_arc.source.name, element.name)
      for out_arc in element.out_arcs:
        G.add_node(out_arc.target.name)
        G.add_edge(element.name, out_arc.target.name)
    return G

  woflan_object.set_s_c_net(woflan_object.get_net() if already_short_circuited else
                              short_circuit_petri_net(woflan_object.get_net(), print_diagnostics=woflan_object.print_diagnostics))
  if woflan_object.get_s_c_net() == None:
    return False
  to_discover = woflan_object.get_s_c_net().places | woflan_object.get_s_c_net().transitions
  graph = transform_petri_net_into_regular_graph(to_discover)
  if not nx.algorithms.components.is_strongly_connected(graph):
    if woflan_object.print_diagnostics:
      print('Petri Net is a not a worflow net.')
    return False
  else:
    if woflan_object.print_diagnostics:
      print("Petri Net is a workflow net.")
    return True #step_3_(woflan_object, return_asap_when_unsound=return_asap_when_unsound)


def step_3_(woflan_object, return_asap_when_unsound=False):
  woflan_object.set_place_invariants(compute_place_invariants(woflan_object.get_s_c_net()))
  woflan_object.set_uniform_place_invariants(transform_basis(woflan_object.get_place_invariants(), style='uniform'))
  woflan_object.set_s_components(
    compute_s_components(woflan_object.get_s_c_net(), woflan_object.get_uniform_place_invariants()))
  woflan_object.set_uncovered_places_s_component(
    compute_uncovered_places_in_component(woflan_object.get_s_components(), woflan_object.get_s_c_net()))
  if len(woflan_object.get_uncovered_places_s_component()) == 0:
    woflan_object.set_left(True)
    if woflan_object.print_diagnostics:
      print('Every place is covered by s-components.')
    return True
  else:
    if woflan_object.print_diagnostics:
      print('The following places are not covered by an s-component: {}.'.format(
        woflan_object.get_uncovered_places_s_component()))
    if return_asap_when_unsound:
      return False
    return False
