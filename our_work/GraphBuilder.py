from EventLog import EventLog
import torch
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from PetriNet import PetriNet
from itertools import combinations

class GraphBuilder:
    def __init__(self):
        pass
    
    def build_petrinet_graph(self, eventlog: EventLog):
        """ Build a Petri net graph from an event log """
        
        # Initialize graph data structure
        graph = Data()
        
        data = {
            'nodes': [],
            'edges': [],
            'node_x': [],
            'node_types': [],
            'labels': []
        }
        
        # Add all activities as transitions
        for activity in eventlog.get_all_activities():
            data['nodes'].append(activity)
            data['node_x'].append(torch.tensor([1.0]))  # Example of transition feature
            data['node_types'].append("transition")
            data['labels'].append(-1)
        
        
        # add candidate places
        data = self.add_candidate_places(data, eventlog)
        
        # Assign each item from the dictionary to the graph
        for key, value in data.items():
            graph[key] = value
        
        graph.edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
                
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
        data = self.add_many_to_many_candidate_places(data, eventlog)
        return data
    
    def add_one_to_one_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add one-to-one candidate places to the Petri net graph based on the event log 
            This methods create a new place for each pair of activities that has a eventually follows relation in the event log.
        """
        footprint_matrix = eventlog.get_footprint_matrix(length=1)
        # footprint matrix look like this {('B', 'A'): '<', ('B', 'C'): '||'}
        for key, value in footprint_matrix.items():
            source, target = key[0], key[1]
            if value == ">":
                # Create a new place
                place_name = f"{source}->{target}"
                
                # Add the place to the data dict
                self.add_place_node(data, place_name)
                
                # Add an edge from the source to the place
                source_index = data['nodes'].index(source)
                place_index = data['nodes'].index(place_name)
                data['edges'].append((source_index, place_index))
                
                # Add an edge from the place to the target
                target_index = data['nodes'].index(target)
                data['edges'].append((place_index, target_index))
        
        return data
    
    def add_one_to_many_candidate_places(self, data: dict, eventlog: EventLog):
        """Add one-to-many candidate places to the Petri net graph based on the event log.
        This method creates a new place if two or more one-to-one places share the same target
        and if all relations between the sources and target are parallel ('||') or choice ('#').
        """
        incoming_edges = {}  # {place: [source1, source2, ...]}
        outgoing_edges = {}  # {place: [target1, target2, ...]}
        
        for edge in data['edges']:
            source, target = edge
            
            # Map incoming edges to each place
            if target not in incoming_edges:
                incoming_edges[target] = []
            incoming_edges[target].append(source)
            
            # Map outgoing edges to each place
            if source not in outgoing_edges:
                outgoing_edges[source] = []
            outgoing_edges[source].append(target)

        same_source_dict = {}  # {source: [target1, target2, ...]}
        same_target_dict = {}  # {target: [source1, source2, ...]}

        # Populate same_source_dict and same_target_dict
        for place in incoming_edges:
            if place in outgoing_edges:
                source_transition = incoming_edges[place][0]
                target_transition = outgoing_edges[place][0]
                same_target_dict.setdefault(target_transition, []).append(source_transition)
                same_source_dict.setdefault(source_transition, []).append(target_transition)
            
        footprint_matrix = eventlog.get_footprint_matrix()
        
        # One to many with the same_source_dict
        for source_transition, target_transitions in same_source_dict.items():
            if len(target_transitions) > 1:
                for transition_combination in self.set_combinations(target_transitions):
                    # check if all the target transistion have # or || relation
                    all_parallel_or_choice = self.check_parallel_or_choice(data, footprint_matrix, transition_combination) 
                    if not all_parallel_or_choice:
                        continue
                    
                    # add candidate to petri net
                    place_name = self.construct_place_name(data, [source_transition], transition_combination)
                    self.add_place_node(data, place_name)
                    self.add_edges(place_name, data, [source_transition], transition_combination)
                
        
        # Many to one with the same_target_dict
        for target_transition, source_transitions in same_target_dict.items():
            if len(source_transitions) > 1:
                for transition_combination in self.set_combinations(source_transitions):
                    # check if all the target transistion have # or || relation
                    all_parallel_or_choice = self.check_parallel_or_choice(data, footprint_matrix, transition_combination) 
                    if not all_parallel_or_choice:
                        continue
                    
                    # add candidate to petri net
                    place_name = self.construct_place_name(data, transition_combination, [target_transition])
                    self.add_place_node(data, place_name)
                    self.add_edges(place_name, data, transition_combination, [target_transition])
                    
        return data

    def add_many_to_many_candidate_places(self, data: dict, eventlog: EventLog):
        """ Add many-to-many candidate places to the Petri net graph based on the event log """
        
        places = [i for i, _ in enumerate(data['nodes']) if data['node_types'][i] == "place"]
        incoming_edges = {}  # Maps each place to its source transitions
        outgoing_edges = {}  # Maps each place to its target transitions
        
        for edge in data['edges']:
            source, target = edge
            if target in places:
                incoming_edges.setdefault(target, []).append(source)
            if source in places:
                outgoing_edges.setdefault(source, []).append(target)
        
        # Populate same_source_dict and same_target_dict
        same_source_dict = {}
        same_target_dict = {}
        for place in places:
            source_transitions = incoming_edges.get(place, [])
            target_transitions = outgoing_edges.get(place, [])
            
            if len(target_transitions) > 1:
                same_source_dict[source_transitions[0]] = target_transitions
            else:
                same_target_dict[target_transitions[0]] = source_transitions
        
        # Build candidate places for many-to-many relationships
        candidate_many_to_many_places = []
        seen_candidates = set() 
        
        for source, targets in same_source_dict.items():
            matching_sources = {source}
            target_set = frozenset(targets)
            
            # Find matching sources that share the same target set
            for other_source, other_targets in same_source_dict.items():
                if source != other_source and target_set == frozenset(other_targets):
                    matching_sources.add(other_source)
            
            # Add to candidate list if unique
            if (frozenset(matching_sources), target_set) not in seen_candidates:
                candidate_many_to_many_places.append((matching_sources, targets))
                seen_candidates.add((frozenset(matching_sources), target_set))
        
        # Check relations and create places for valid subsets
        footprint_matrix = eventlog.get_footprint_matrix()
        for sources, targets in candidate_many_to_many_places:
            for transition_combination in self.set_combinations(sources):
                if self.check_parallel_or_choice(data, footprint_matrix, transition_combination):
                    # Construct and add the place node and edges
                    place_name = self.construct_place_name(data, transition_combination, targets)
                    self.add_place_node(data, place_name)
                    self.add_edges(place_name, data, transition_combination, targets)
        
        return data
    
    def set_combinations(self, transitions: list):
        """ Set the combinations of sources or targets """
        set_combinations = []
        for i in range(2, len(transitions) + 1):
            set_combinations.extend(combinations(transitions, i))
        return set_combinations
    
    def check_parallel_or_choice(self, data, footprint_matrix: dict, transitions: list):
        """ Check if the relation between source and target is parallel ('||') or choice ('#') """
        for i, t in enumerate(transitions, 0):
            for t2 in transitions[i + 1:]:
                entry = footprint_matrix[(data['nodes'][t], data['nodes'][t2])]
                if entry != '||' and entry != '#':
                    return False
        return True
    
    def construct_place_name(self, data, sources, targets):
        """ Construct the name of the place based on the source and target transitions """
        place_name = ",".join(data['nodes'][activity] for activity in sources)
        place_name += "->" + ",".join(data['nodes'][activity] for activity in targets)
        return place_name
    
    def add_place_node(self, data, place_name):
        """ Add a place node to the data dict"""
        data['nodes'].append(place_name)
        data['node_x'].append(torch.tensor([1.0]))
        data['node_types'].append("place")
        data['labels'].append(-1)
        
    def add_edges(self, place_name: str, data, sources: list, targets: list):
        """Adds edges between candidate place and source/target transitions"""
        place_index = data['nodes'].index(place_name)
        for source in sources:
            data['edges'].append((source, place_index))
        for target in targets:
            data['edges'].append((place_index, target))
            
    def annotate_petrinet_graph(self, graph: Data, petrinet: PetriNet):
        """
        Annotate the graph with the petrinet information populates the labels of the nodes and edges of the graph with the 1, 0, -1 values. 
        1 if the node is a true place, 0 if the node is not a true place, -1 if the node is a transition.
        """
        
        place_mask = [i for i, _ in enumerate(graph['nodes']) if graph['node_types'][i] == "place"]
        
        # for each place in the graph, check if it is a true place or not
        for i in place_mask:
            graph_place = graph['nodes'][i]
            graph_ingoing_edges = graph['edge_index'][0][graph['edge_index'][1] == i]
            graph_outgoing_edges = graph['edge_index'][1][graph['edge_index'][0] == i]
            
            
            ingoing_transitions = [graph['nodes'][j] for j in graph_ingoing_edges if graph['node_types'][j] == "transition"]
            outgoing_transitions = [graph['nodes'][j] for j in graph_outgoing_edges if graph['node_types'][j] == "transition"]
            
            is_place = petrinet.is_place(ingoing_transitions, outgoing_transitions)
            
            if is_place:
                graph['labels'][i] = 1
            else:
                graph['labels'][i] = 0
            
        return graph
        