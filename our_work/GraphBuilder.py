from EventLog import EventLog
import torch
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

class GraphBuilder:
    def __init__(self):
        pass
    
    def build_petrinet_graph(self, eventlog: EventLog):
        """ Build a Petri net graph from an event log """
        
        # Initialize graph data structure
        graph = HeteroData()

        # Prepare data in a dictionary for unpacking
        data = {
            'transitions': [],
            'transitions_x': [], 
            'candidate_places': [],
            'candidate_places_x': [],
            'candidate_types': [],
            ('transition', 'to', 'place'): [],
            ('place', 'to', 'transition'): []
        }
        
        # Add all activities as transitions
        for activity in eventlog.get_all_activities():
            data['transitions'].append(activity)
            data['transitions_x'].append(torch.tensor([1.0]))  # Example of transition feature
        
        
        # add candidate places
        data = self.add_candidate_places(data, eventlog)
        

        # Assign each item from the dictionary to the graph
        for key, value in data.items():
            graph[key] = value
        
        
        return graph
        
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
    def visualize_trace_graph(data: Data):
        """Visualize a PyG graph using Matplotlib and NetworkX."""
        # Convert PyG data to a NetworkX graph
        G = to_networkx(data, to_undirected=False)

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
    
    def add_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add candidate places to the Petri net graph based on the event log """
        data = self.add_one_to_one_candidate_places(data, eventlog)
        data = self.add_one_to_many_candidate_places(data, eventlog)
        data = self.add_many_to_one_candidate_places(data, eventlog)
        data = self.add_many_to_many_candidate_places(data, eventlog)
        return data
    
    def add_one_to_one_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add one-to-one candidate places to the Petri net graph based on the event log 
            This methods create a new place for each pair of activities that are directly followed by each other in the event log.
        """
        footprint_matrix = eventlog.get_footprint_matrix()
        # footprint matrix look like this {('B', 'A'): '<', ('B', 'C'): '||'}
        for key, value in footprint_matrix.items():
            source, target = key[0], key[1]
            if value == ">":
                # Create a new place
                place_name = f"{source}->{target}"
                data['candidate_places'].append(place_name)
                data['candidate_places_x'].append(torch.tensor([1.0]))
                # Add edges from source to place and place to target
                transition_index = data['transitions'].index(source)
                place_index = data['candidate_places'].index(place_name)
                data[('transition', 'to', 'place')].append((transition_index, place_index))
                data[('place', 'to', 'transition')].append((place_index, data['transitions'].index(target)))
        
        return data
    
    def add_one_to_many_candidate_places(self, data: dict, eventlog: EventLog):
        """Add one-to-many candidate places to the Petri net graph based on the event log.
        This method creates a new place if two or more one-to-one places share the same target
        and if all relations between the sources and target are parallel ('||').
        """
        # find all the places in the data dict that have the same target 
        places_with_same_source = {} # {source: [target1, target2, ...]} all activites
        for arc in data[('place', 'to', 'transition')]:
            target_transition_index = arc[1]
            target_transition = data['transitions'][target_transition_index]
            source_candidate_index = arc[0]
            source_candidate = data['candidate_places'][source_candidate_index]
            
            # find the transition that corresponds to the source candidate
            source_transition_index = data['transitions'].index(source_candidate.split("->")[0])
            source_transition = data['transitions'][source_transition_index]
            
            if source_transition not in places_with_same_source:
                places_with_same_source[source_transition] = []
            places_with_same_source[source_transition].append(target_transition)
            
            
        footprint_matrix = eventlog.get_footprint_matrix()
        print(f"Places with the same target: {places_with_same_source}")
        # check if the same source transitions have the same target transition
        for source_transition, target_transitions in places_with_same_source.items():
            if len(target_transitions) > 1:
                # check if all the source transitions have the same target transition
                targets = target_transitions
                source = source_transition
                # check if all the source transitions have the same target transition
                all_parallel = True
                for target in targets:
                    for target1 in targets:
                        if target != target1:
                            if footprint_matrix[(target, target1)] != '||' and footprint_matrix[(target1, target)] != '#':
                                all_parallel = False
                                break
                if all_parallel:
                    # Create a new place
                    place_name = f"{source}->"
                    for target in targets:
                        place_name += f"{target},"
                    place_name = place_name[:-1]
                    data['candidate_places'].append(place_name)
                    data['candidate_places_x'].append(torch.tensor([1.0]))
                    
                    # add an edge from the transition to the place
                    source_index = data['transitions'].index(source)
                    place_index = data['candidate_places'].index(place_name)
                    data[('transition', 'to', 'place')].append((source_index, place_index))
                    
                    # add an edge from the place to all the target transitions
                    for target in targets:
                        target_index = data['transitions'].index(target)
                        data[('place', 'to', 'transition')].append((place_index, target_index))

                    
        return data
             
    def add_many_to_one_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add many-to-one candidate places to the Petri net graph based on the event log """
        # find all the places in the data dict that have the same target 
        places_with_same_target = {} # {target: [source1, source2, ...]} all activites
        for arc in data[('transition', 'to', 'place')]:
            source_transition_index = arc[0]
            source_transition = data['transitions'][source_transition_index]
            target_candidate_index = arc[1]
            target_candidate = data['candidate_places'][target_candidate_index]
            
            # check the name of the target candidate if it has more than one source by checking if it follows the pattern "source->target"
            if target_candidate.split("->")[1] not in data['transitions']:
                continue
            
            
            # find the transition that corresponds to the source candidate
            target_transition_index = data['transitions'].index(target_candidate.split("->")[1])
            target_transition = data['transitions'][target_transition_index]
            
            if target_transition not in places_with_same_target:
                places_with_same_target[target_transition] = []
            places_with_same_target[target_transition].append(source_transition)
            
            
        footprint_matrix = eventlog.get_footprint_matrix()
        print(f"Places with the same target: {places_with_same_target}")
        # check if the same source transitions have the same target transition
        for source_transition, target_transitions in places_with_same_target.items():
            if len(target_transitions) > 1:
                # check if all the source transitions have the same target transition
                targets = target_transitions
                source = source_transition
                # check if all the source transitions have the same target transition
                all_parallel = True
                for target in targets:
                    for target1 in targets:
                        if target != target1:
                            if footprint_matrix[(target, target1)] != '||' and footprint_matrix[(target1, target)] != '#':
                                all_parallel = False
                                break
                if all_parallel:
                    # Create a new place
                    place_name = f"{source}->"
                    for target in targets:
                        place_name += f"{target},"
                    place_name = place_name[:-1]
                    data['candidate_places'].append(place_name)
                    data['candidate_places_x'].append(torch.tensor([1.0]))
                    
                    # add an edge from the transition to the place
                    source_index = data['transitions'].index(source)
                    place_index = data['candidate_places'].index(place_name)
                    data[('transition', 'to', 'place')].append((source_index, place_index))
                    
                    # add an edge from the place to all the target transitions
                    for target in targets:
                        target_index = data['transitions'].index(target)
                        data[('place', 'to', 'transition')].append((place_index, target_index))

                    
        return data
    
    def add_many_to_many_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add many-to-many candidate places to the Petri net graph based on the event log """
        return data