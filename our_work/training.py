import torch
import time
from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from Models import GNNWithClassifier
from util import batch_eventlog_loader, batch_petrinet_loader
from torch_geometric.loader import DataLoader

def train(eventlog_dir, petrinet_dir, input_dim=64, hidden_dim=16, dense_hidden_dim=32, output_dim=1, 
          lr=0.01, epochs=100, batch_size=10, cpu_count=1):
    # Set device to GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    print(f"Training on device: {device}")

    # Instantiate the model, optimizer, and loss function
    model = GNNWithClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dense_hidden_dim=dense_hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))

    start_time = time.time()  # Start timing the entire training process

    for epoch in range(epochs):
        epoch_loss = 0  # Track loss for each epoch
        model.train()
        epoch_start = time.time()

        # Iterate through batches of EventLogs and PetriNets
        for eventlog_batch, petrinet_batch in zip(batch_eventlog_loader(eventlog_dir, batch_size, cpu_count),
                                                  batch_petrinet_loader(petrinet_dir, batch_size, cpu_count)):
            pyg_graphs = []

            # Build and annotate graphs for the current batch
            for id, petrinet in petrinet_batch.items():
                if id in eventlog_batch:  # Ensure matching EventLog exists
                    eventlog = eventlog_batch[id]
                    graph_builder = GraphBuilder(eventlog)
                    graph = graph_builder.build_petrinet_graph()
                    graph = graph_builder.annotate_petrinet_graph(graph, petrinet)
                    pyg_graphs.append(graph)
            # Create a DataLoader for the current batch of graphs
            data_loader = DataLoader(pyg_graphs, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)

            # Train on the current batch
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                out = model(batch.node_x, batch.edge_index)
                
                # Compute loss only for place nodes
                loss = criterion(out[batch['place_mask']], batch.labels[batch["place_mask"]])

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # Print average loss per epoch
        avg_loss = epoch_loss / (len(petrinet_batch) * batch_size)
        epoch_end = time.time()
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Epoch time: {epoch_end - epoch_start:.2f}s")

    # Calculate the total training time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Export the model after training
    torch.save(model.state_dict(), "models/graph_sage_model_with_dense_classifier.pth")
    print("Model saved as 'graph_sage_model_with_dense_classifier.pth'")

if __name__ == "__main__":
    train(eventlog_dir="./data_generation/synthetic_data",
          petrinet_dir="./data_generation/synthetic_data",
          epochs=1, batch_size=10, cpu_count=4)

        
    
