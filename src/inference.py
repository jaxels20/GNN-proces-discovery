
from src.EventLog import EventLog
from src.Models import GNNWithClassifier
from src.GraphBuilder import GraphBuilder
import torch
from torch_geometric.data import Data
from src.PetriNet import PetriNet
import copy


def do_inference(graph: Data, model: torch.nn.Module, device: torch.device):
    """
    Perform inference on the graph using the given model and device (CPU/GPU).

    Args:
        graph (Data): A PyTorch Geometric Data object representing the graph.
        model (torch.nn.Module): The trained model.
        device (torch.device): The device to use (e.g., torch.device('cuda') or torch.device('cpu')).

    Returns:
        Data: The updated graph with predictions.
    """
    # Move the graph data to the specified device
    graph.node_x = graph.node_x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.place_mask = graph.place_mask.to(device)
    graph.selected_nodes = graph.selected_nodes.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        # Perform the forward pass
        out = model(graph.node_x, graph.edge_index)

    # Apply thresholding for binary classification
    predicted = out > 0.5

    # Update the selected nodes for places (place_mask)
    graph.selected_nodes[graph.place_mask] = predicted[graph.place_mask].squeeze()

    return graph


    
    



    
    
    