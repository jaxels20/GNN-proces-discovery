from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc

from models import GraphSAGEModel

import torch

if __name__ == "__main__":
    log_file_name = "./example/many_to_many.xes"
    pn_file_name = "./example/many_to_many.ptml"
    eventlog = EventLog.load_xes(log_file_name)
    pn = PetriNet.from_ptml(pn_file_name)
    
    gb = GraphBuilder()
    graph = gb.build_petrinet_graph(eventlog)
    
    graph = gb.annotate_petrinet_graph(graph, pn)
    
    
    # Instantiate model, optimizer, and loss function
    model = GraphSAGEModel(input_dim=1, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(graph.node_x, graph.edge_index)
        
        # Compute loss only for place nodes
        loss = criterion(out[graph['place_mask']], graph.labels[graph["place_mask"]])

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")







