import os
from EventLog import EventLog
from PetriNet import PetriNet
from GraphBuilder import GraphBuilder
from torch_geometric.data import Data
import torch

def select_all_places(graph: Data) -> Data:
    """set selected_nodes to true for all nodes in the graph
        and return the graph
    """
    graph["selected_nodes"] = torch.ones(graph.num_nodes, dtype=torch.bool)
    return graph

def compare_candidate_places_to_true_places(candidate_pn: PetriNet, true_pn: PetriNet):
    """compare the places of the candidate_pn to the places of the true_pn
        and return true positives, false positives, false negatives
    """
    
    true_places = true_pn.places
    candidate_places = candidate_pn.places
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for candidate_place in candidate_places:
        found_true_positive = False
        candidate_ingoing_transitions = sorted(candidate_pn.get_ingoing_transitions(candidate_place.name))
        candidate_outgoing_transitions = sorted(candidate_pn.get_outgoing_transitions(candidate_place.name))
        for true_place in true_places:
            true_ingoing_transitions = sorted(true_pn.get_ingoing_transitions(true_place.name))
            true_outgoing_transitions = sorted(true_pn.get_outgoing_transitions(true_place.name))
            if candidate_ingoing_transitions == true_ingoing_transitions and candidate_outgoing_transitions == true_outgoing_transitions:
                true_positives += 1
                found_true_positive = True
        if not found_true_positive:
            false_positives += 1
                
    # do the same for false negatives
    for true_place in true_places:
        found_false_negative = True
        true_ingoing_transitions = sorted(true_pn.get_ingoing_transitions(true_place.name))
        true_outgoing_transitions = sorted(true_pn.get_outgoing_transitions(true_place.name))
        for candidate_place in candidate_places:
            candidate_ingoing_transitions = sorted(candidate_pn.get_ingoing_transitions(candidate_place.name))
            candidate_outgoing_transitions = sorted(candidate_pn.get_outgoing_transitions(candidate_place.name))
            if candidate_ingoing_transitions == true_ingoing_transitions and candidate_outgoing_transitions == true_outgoing_transitions:
                found_false_negative = False
        if found_false_negative:
            false_negatives += 1


    return true_positives, false_positives, false_negatives
    

    


if __name__ == "__main__":
    eventlog = EventLog.load_xes("./controlled_scenarios/simple_and_split/eventlog.xes")
    petrinet = PetriNet.from_ptml("./controlled_scenarios/simple_and_split/petri_net.ptml")
    graphbuilder = GraphBuilder(eventlog)
    graph = graphbuilder.build_petrinet_graph()
    graph = select_all_places(graph)
    candidate_pn = PetriNet.from_graph(graph)
    result = compare_candidate_places_to_true_places(candidate_pn, petrinet)
    print(result)