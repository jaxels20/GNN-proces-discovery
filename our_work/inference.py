
from EventLog import EventLog
from Models import GraphSAGEModel
from GraphBuilder import GraphBuilder
import torch
from torch_geometric.data import Data
from PetriNet import PetriNet


def do_inference(graph: Data, model: torch.nn.Module):
    model.eval()
    out = model(graph.node_x, graph.edge_index)
    # if a value is greater than 0.5, it is a place
    predicted = out > 0.5
    graph['selected_nodes'][graph['place_mask']] = predicted[graph['place_mask']].squeeze()
    
    return graph
    
def discover(eventlog: EventLog, model_path: str = "./models/graph_sage_model.pth"):
    # Load the model
    model = GraphSAGEModel(input_dim=64, hidden_dim=16, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    
    # Get the event log as a graph
    graph_builder = GraphBuilder(eventlog=eventlog)
    graph = graph_builder.build_petrinet_graph()
    
    graph = do_inference(graph, model)
    
    discovered_pn = PetriNet.from_graph(graph)
    
    return discovered_pn
    
if __name__ == "__main__":
    eventlog = EventLog.load_xes("./synthetic_data/log_3.xes")
    discover(eventlog, "./models/graph_sage_model.pth")
    
    



    
    
    