from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc

from Models import GraphSAGEModel
from util import load_all_petrinets, load_all_eventlogs
import torch


def train(all_graphs, input_dim=64, hidden_dim=16, output_dim=1, lr=0.01, epochs=100):
    # Instantiate model, optimizer, and loss function
    model = GraphSAGEModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        epoch_loss = 0  # Track loss for all graphs in each epoch
        model.train()
        
        for graph in all_graphs:
            optimizer.zero_grad()

            # Forward pass
            out = model(graph.node_x, graph.edge_index)

            # Compute loss only for place nodes
            loss = criterion(out[graph['place_mask']], graph.labels[graph["place_mask"]])

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # Print average loss per epoch
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(all_graphs)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss}")

    # Export the model after training
    torch.save(model.state_dict(), "models/graph_sage_model.pth")
    print("Model saved as 'graph_sage_model.pth'")
    
        
if __name__ == "__main__":
    all_petrinets = load_all_petrinets("./synthetic_data")
    all_eventlogs = load_all_eventlogs("./synthetic_data")
    
    pyg_graphs = []
    
    for id, petrinet in all_petrinets.items():
        eventlog = all_eventlogs[id]
        graph_builder = GraphBuilder(eventlog)
        graph = graph_builder.build_petrinet_graph()
        graph = graph_builder.annotate_petrinet_graph(graph, petrinet)
        pyg_graphs.append(graph)
    
    train(pyg_graphs, epochs=100)
        
        

    
    





