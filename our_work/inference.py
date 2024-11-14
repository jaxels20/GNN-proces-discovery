
from EventLog import EventLog
from Models import GNNWithClassifier
from GraphBuilder import GraphBuilder
import torch
from torch_geometric.data import Data
from PetriNet import PetriNet
import copy


def do_inference(graph: Data, model: torch.nn.Module):
    model.eval()
    out = model(graph.node_x, graph.edge_index)
    # if a value is greater than 0.5, it is a place
    predicted = out > 0.5
    graph['selected_nodes'][graph['place_mask']] = predicted[graph['place_mask']].squeeze()
    
    return graph
    


    
    



    
    
    