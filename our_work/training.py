import torch
import time
from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from Models import GNNWithClassifier
from util import load_all_petrinets, load_all_eventlogs
from torch_geometric.loader import DataLoader

def train(all_graphs, input_dim=64, hidden_dim=16, dense_hidden_dim=32, output_dim=1, lr=0.01, epochs=100):
    # Set device to GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    print(f"Training on device: {device}")

    # Instantiate the new combined model, optimizer, and loss function
    model = GNNWithClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dense_hidden_dim=dense_hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))
    data_loader = DataLoader(all_graphs, batch_size=150, shuffle=True, pin_memory=True, num_workers=4)
    
    start_time = time.time()  # Start timing the entire training loop

    for epoch in range(epochs):
        epoch_loss = 0  # Track loss for all graphs in each epoch
        model.train()

        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(data_loader):
            batch_loading_start = time.time()
            
            # Move batch to the GPU
            batch = batch.to(device)
            batch_loading_end = time.time()
            
            optimizer.zero_grad()

            # Forward pass
            forward_start = time.time()
            out = model(batch.node_x, batch.edge_index)  # Uses the combined model for feature extraction and classification
            forward_end = time.time()

            # Compute loss only for place nodes
            loss_start = time.time()
            loss = criterion(out[batch['place_mask']], batch.labels[batch["place_mask"]])
            loss_end = time.time()

            # Backpropagation
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_end = time.time()

            epoch_loss += loss.item()
            
            # Print timing info for each batch (optional, for debugging)
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}:")
                print(f"  Batch loading time: {batch_loading_end - batch_loading_start:.4f}s")
                print(f"  Forward pass time: {forward_end - forward_start:.4f}s")
                print(f"  Loss computation time: {loss_end - loss_start:.4f}s")
                print(f"  Backward pass time: {backward_end - backward_start:.4f}s")

        # Print average loss per epoch and epoch timing
        avg_loss = epoch_loss / len(all_graphs)
        epoch_end = time.time()
        print(f"Epoch {epoch}, Avg Loss: {avg_loss}, Epoch time: {epoch_end - epoch_start:.2f}s")

    # Calculate the total training time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Export the model after training
    torch.save(model.state_dict(), "models/graph_sage_model_with_dense_classifier.pth")
    print("Model saved as 'graph_sage_model_with_dense_classifier.pth'")
    
        
if __name__ == "__main__":
    all_petrinets = load_all_petrinets("./data_generation/synthetic_data")
    all_eventlogs = load_all_eventlogs("./data_generation/synthetic_data")
    
    pyg_graphs = []
    
    for id, petrinet in all_petrinets.items():
        eventlog = all_eventlogs[id]
        graph_builder = GraphBuilder(eventlog)
        graph = graph_builder.build_petrinet_graph()
        graph = graph_builder.annotate_petrinet_graph(graph, petrinet)
        pyg_graphs.append(graph)
    
    train(pyg_graphs, epochs=100)


        
        

    
    





