from EventLog import EventLog
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

class GraphBuilder:
    def __init__(self):
        pass
    
    def build_petrinet_graph(self, eventlog: EventLog):
        """ Build a Petri net graph from an event log """
        pass
    
    @staticmethod
    def build_trace_graph(eventlog: EventLog):
        """ Build a trace graph from an event log with a single start and end node using PyTorch Geometric. """
        # Step 1: Initialize lists for nodes and edges
        node_list = []  # List of node features
        edge_index = []  # List of edges (source, target)
        node_names = []  # List of node names
        
        # Add artificial start node (index 0) and end node (index 1)
        start_node_index = 0
        end_node_index = 1
        
        # Add start node
        node_list.append([1.0]*5)  # You can customize the feature of the start node
        node_names.append("Start")  # Name of the start node
        # Add end node
        node_list.append([1.0]*5)  # You can customize the feature of the end node
        node_names.append("End")  # Name of the end node

        # Step 2: Iterate through traces in the event log
        for trace in eventlog.traces:
            # Create a node for each event in the trace
            event_indices = []  # To keep track of the indices of the current trace events
            for event in trace.events:
                event_index = len(node_list)  # Current event index
                # Example of using one-hot encoding for the activity as node features
                activity_features = [1.0 if activity == event.activity else 0.0 for activity in eventlog.get_all_activities()]
                node_list.append(activity_features)  # Add event node
                node_names.append(event.activity)  # Name of the event node
                event_indices.append(event_index)

            # Step 3: Create edges from start to the first event and from the last event to end
            edge_index.append((start_node_index, event_indices[0]))  # Start -> First event
            edge_index.append((event_indices[-1], end_node_index))  # Last event -> End
            
            # Step 4: Create edges between events in the trace
            for i in range(len(event_indices) - 1):
                edge_index.append((event_indices[i], event_indices[i + 1]))  # Event[i] -> Event[i + 1]
        
        # Convert to tensor format for PyG
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
        x = torch.tensor(node_list, dtype=torch.float)  # Node features tensor

        # Step 5: Create a PyG Data object
        data = Data(edge_index=edge_index, x=x, node_names=node_names)
        return data
    
    @staticmethod
    def visualize_graph(data: Data):
        """Visualize a PyG graph using Matplotlib and NetworkX."""
        # Convert PyG data to a NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Step 2: Add node names as attributes
        for i, name in enumerate(data.node_names):
            G.nodes[i]['name'] = name

        # Step 3: Draw the graph with node names
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=False, node_size=700)

        # Draw node labels
        labels = nx.get_node_attributes(G, 'name')
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # Show the plot
        plt.title("Graph Visualization with Node Names")
        plt.show()
        
        
        