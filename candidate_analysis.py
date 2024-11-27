import os
from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.GraphBuilder import GraphBuilder
from src.Comparison import compare_discovered_pn_to_true_pn
from torch_geometric.data import Data
import torch

INPUT_DIR = "./controlled_scenarios/" # Assume structered like this "./controlled_scenarios/dataset_name/"
OUTPUT_DIR = "./con_scenarios_candidate_analysis_results/" 

def select_all_places(graph: Data) -> Data:
    """set selected_nodes to true for all nodes in the graph
        and return the graph
    """
    graph["selected_nodes"] = torch.ones(graph.num_nodes, dtype=torch.bool)
    return graph


if __name__ == "__main__":
    eventlog = EventLog.load_xes("./controlled_scenarios/loop_lenght_2/eventlog.xes")
    petrinet = PetriNet.from_ptml("./controlled_scenarios/loop_lenght_2/petri_net.ptml")
    graphbuilder = GraphBuilder(eventlog)
    graph = graphbuilder.build_petrinet_graph()
    graph = select_all_places(graph)
    candidate_pn = PetriNet.from_graph(graph)
    result = compare_discovered_pn_to_true_pn(candidate_pn, petrinet)
    print(result)