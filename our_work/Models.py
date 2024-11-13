import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# 1. GraphSAGE feature extractor class
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # Apply GraphSAGE layers for feature extraction
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

# 2. Dense classifier class
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply dense layers for classification
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # For binary classification
        return x

# 3. Combined model that uses both GraphSAGE feature extractor and Dense classifier
class GNNWithClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_hidden_dim, output_dim):
        super(GNNWithClassifier, self).__init__()
        # Instantiate the GraphSAGE feature extractor and dense classifier
        self.feature_extractor = GNN(input_dim, hidden_dim)
        self.classifier = Classifier(hidden_dim, dense_hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Pass data through GraphSAGE feature extractor
        x = self.feature_extractor(x, edge_index)
        # Pass extracted features through dense classifier
        x = self.classifier(x)
        return x