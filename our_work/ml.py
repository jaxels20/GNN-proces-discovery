



import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GraphSAGE layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x


# Prepare the model, optimizer, and loss function
model = GraphSAGENet(in_channels=2, hidden_channels=16, out_channels=1)  # Adjust hidden size as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Labels for places
data.y = torch.tensor(toy_data['true_places'], dtype=torch.float)

# Training loop
for epoch in range(100):  # Adjust epochs as needed
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data).squeeze()
    
    # Compute loss and backpropagate
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")







