import random
from graphviz import Digraph
from pm4py.objects.petri_net.obj import PetriNet as PM4PyPetriNet, Marking
from pm4py.analysis import check_soundness
from pm4py.objects.petri_net.utils.check_soundness import (
    check_easy_soundness_net_in_fin_marking,
)
from pm4py.objects.process_tree.importer.variants.ptml import apply as import_ptml_tree
from pm4py.objects.conversion.process_tree.variants.to_petri_net import apply as convert_pt_to_pn
from src.EventLog import EventLog
from pm4py.write import write_ptml
from pm4py.convert import convert_to_process_tree as convert_to_pt
from torch_geometric.data import Data
from copy import deepcopy

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import figure_generation.constants as constants


class Place:
    """
    Class representing a place in a Petri net.

    Attributes:
    -----------
    name : str
        The name of the place.
    tokens : int
        The number of tokens in the place.
    """

    def __init__(self, name: str, tokens: int = 0):
        self.name = name
        self.tokens = tokens

    def add_tokens(self, count: int = 1):
        """Add tokens to the place."""
        self.tokens += count

    def remove_tokens(self, count: int = 1):
        """Remove tokens from the place, ensuring no negative token count."""
        if self.tokens - count < 0:
            raise ValueError(
                f"Cannot remove {count} tokens from place '{self.name}' (tokens = {self.tokens})"
            )
        self.tokens -= count

    def __repr__(self):
        return f"Place({self.name}, tokens={self.tokens})"

class Transition:
    """
    Class representing a transition in a Petri net.

    Attributes:
    -----------
    name : str
        The name of the transition.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Transition({self.name})"
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __gt__(self, other):
        return self.name > other.name
    
    def __lt__(self, other):
        return self.name < other.name

class Arc:
    """
    Class representing an arc in a Petri net, connecting a place and a transition.

    Attributes:
    -----------
    source : Place or Transition
        The source of the arc (either a Place or a Transition).
    target : Place or Transition
        The target of the arc (either a Place or a Transition).
    weight : int
        The weight of the arc.
    """

    def __init__(self, source: str, target: str, weight: int = 1):
        self.source = source
        self.target = target
        self.weight = weight

    def __repr__(self):
        return f"Arc({self.source} -> {self.target}, weight={self.weight})"


class PetriNet:
    """
    Class representing a generic Petri net.

    Attributes:
    -----------
    places : list[Place]
        List of places in the Petri net.
    transitions : list[Transition]
        List of transitions in the Petri net.
    arcs : list[Arc]
        List of arcs connecting places and transitions.
    """

    def __init__(self, places: list = [], transitions: list = [], arcs: list = []):
        self.places = places
        self.transitions = transitions
        self.arcs = arcs

    def __repr__(self):
        return f"PetriNet(Places: {len(self.places)}, Transitions: {len(self.transitions)}, Arcs: {len(self.arcs)})"
    
    def add_place(self, name: str, tokens: int = 0):
        """Add a place to the Petri net."""
        if name in [place.name for place in self.places]:
            raise ValueError(f"Place '{name}' already exists in the Petri net")

        place = Place(name, tokens)
        self.places.append(place)

    def add_transition(self, name: str):
        """Add a transition to the Petri net."""
        if name in [transition.name for transition in self.transitions]:
            raise ValueError(f"Transition '{name}' already exists in the Petri net")

        transition = Transition(name)
        self.transitions.append(transition)

    def add_arc(self, source: str, target: str, weight: int = 1):
        """Add an arc connecting a place and a transition or vice versa."""
        ids = [place.name for place in self.places] + [transition.name for transition in self.transitions]
        if source not in ids:
            raise ValueError(f"Source '{source}' does not exist in the Petri net")
        if target not in ids:
            raise ValueError(f"Target '{target}' does not exist in the Petri net")
        
        arc = Arc(source, target, weight)
        self.arcs.append(arc)

    def get_place_by_name(self, name: str):
        """Return a place by its name, or None if it doesn't exist."""
        for place in self.places:
            if place.name == name:
                return place
        return None

    def get_transition_by_name(self, name: str):
        """Return a transition by its name, or None if it doesn't exist."""
        for transition in self.transitions:
            if transition.name == name:
                return transition
        return None

    def get_ingoing_transitions(self, place_name: str) -> list[Transition]:
        """Return the transitions that have an arc to the place."""
        return [self.get_transition_by_name(arc.source) for arc in self.arcs if arc.target == place_name]
    
    def get_outgoing_transitions(self, place_name: str) -> list[Transition]:
        """Return the transitions that have an arc from the place."""
        return [self.get_transition_by_name(arc.target) for arc in self.arcs if arc.source == place_name]

    def is_transition_enabled(self, transition_name: str) -> bool:
        """
        Check if a transition is enabled. A transition is enabled if all its input places have enough tokens.
        """
        input_arcs = [arc for arc in self.arcs if arc.target == transition_name]
        return all(
            self.get_place_by_name(arc.source).tokens >= arc.weight
            for arc in input_arcs
        )

    def fire_transition(self, transition: Transition):
        """
        Fire a transition if it is enabled, moving tokens from input places to output places.
        
        Raises:
        -------
        ValueError : If the transition is not enabled.
        """
        if not self.is_transition_enabled(transition.name):
            raise ValueError(f"Transition '{transition.name}' is not enabled")

        # Remove tokens from input places
        input_arcs = [arc for arc in self.arcs if arc.target == transition.name]
        for arc in input_arcs:
            input_place = self.get_place_by_name(arc.source)
            input_place.remove_tokens(arc.weight)

        # Add tokens to output places
        output_arcs = [arc for arc in self.arcs if arc.source == transition.name]
        for arc in output_arcs:
            output_place = self.get_place_by_name(arc.target)
            output_place.add_tokens(arc.weight)

    def visualize(self, filename="petri_net", format="pdf", random_place_naming=False):
        """
        Visualize the Petri net and save it as a PNG file using Graphviz.

        Parameters:
        -----------
        filename : str
            The base filename for the output file (without extension).
        format : str
            The format for the output file (e.g., 'png', 'pdf').
        """
        dot = self.get_visualization(format=format, random_place_naming=random_place_naming)

        output_path = dot.render(filename, cleanup=True, format=format)
        print(f"Petri net saved as {output_path}")

    def get_visualization(self, format, random_place_naming=False):
        dot = Digraph(comment="Petri Net", format=format)
        curr_place_id = 0
        
        for place in self.places:            
            color = "black"
            style = "rounded"
            
            if not random_place_naming:
                label = place.name
            elif place.name == "start" or place.name == "source":
                label = "Start"
            elif place.name == "end" or place.name == "sink":
                label = "End"
            else:
                label = f"p{curr_place_id}"
                curr_place_id += 1


            if place.tokens > 0:
                label += f"\nTokens: {place.tokens}"

            
            dot.node(
                place.name,
                label=label,
                shape="circle",
                color=color,
                style=style,
            )

        for transition in self.transitions:

            # if the transition is a tau transition, color it black and fill it
            if transition.name.startswith("tau"):
                color = "black"
                style = "filled"
                label = ""
            else:
                # if the transition is not a tau transition, color it black and do not fill it
                color = "black"
                style = ""
                label = transition.name
                
            
            dot.node(
                transition.name,
                label=transition.name,
                shape="box",
                color=color,
                style=style,
            )

        for arc in self.arcs:
            edge_label = "" #str(arc.weight)
            dot.edge(arc.source, arc.target, label=edge_label)

        return dot

    def get_start_place(self):
        """Return the start places (no incoming arcs), or None if none exists."""
        for place in self.places:
            incoming_arcs = [arc for arc in self.arcs if arc.target == place.name]
            if len(incoming_arcs) == 0:
                return place        
        return None
 
    def get_end_place(self):
        """Return the end place (no outgoing arcs), or None if none exists."""
        for place in self.places:
            outgoing_arcs = [arc for arc in self.arcs if arc.source == place.name]
            if len(outgoing_arcs) == 0:
                return place
        return None

    def construct_start_place(self):    
        """Construct a start place with no incoming arcs. Use all the transitions that has no incoming arcs."""
        # check if a start place already exists
        for place in self.places:
            incoming_arcs = [arc for arc in self.arcs if arc.target == place.name]
            if len(incoming_arcs) == 0:
                continue
        
        start_place = Place('start')
        start_place.tokens = 1
        self.places.append(start_place)
        for transition in self.transitions:
            incoming_arcs = [arc for arc in self.arcs if arc.target == transition.name]
            if len(incoming_arcs) == 0:
                self.arcs.append(Arc(start_place.name, transition.name))
        
    def construct_end_place(self):
        """Construct an end place with no outgoing arcs. Use all the transitions that has no outgoing arcs."""
        # check if an end place already exists
        for place in self.places:
            outgoing_arcs = [arc for arc in self.arcs if arc.source == place.name]
            if len(outgoing_arcs) == 0:
                continue
        
        end_place = Place('end')
        self.places.append(end_place)
        for transition in self.transitions:
            outgoing_arcs = [arc for arc in self.arcs if arc.source == transition.name]
            if len(outgoing_arcs) == 0:
                self.arcs.append(Arc(transition.name, end_place.name))

    def to_pm4py(self):
        """Convert our Petri net class to a pm4py Petri net and return it"""
        pm4py_pn = PM4PyPetriNet()
        pm4py_dict = {}

        for place in self.places:
            pm4py_place = PM4PyPetriNet.Place(place.name)
            pm4py_pn.places.add(pm4py_place)
            pm4py_dict[place.name] = pm4py_place

        for transition in self.transitions:
            pm4py_transition = PM4PyPetriNet.Transition(
                transition.name, label=transition.name if not transition.name.startswith("tau") else None
            )
            pm4py_pn.transitions.add(pm4py_transition)
            pm4py_dict[transition.name] = pm4py_transition

        for arc in self.arcs:
            pm4py_arc = PM4PyPetriNet.Arc(
                pm4py_dict[arc.source], pm4py_dict[arc.target]
            )
            pm4py_pn.arcs.add(pm4py_arc)
            
            # Add out arc and in arc property to places
            if arc.source in [p.name for p in pm4py_pn.places]:
                place = pm4py_dict[arc.source]
                place.out_arcs.add(pm4py_arc)
            else:
                place = pm4py_dict[arc.target]
                place.in_arcs.add(pm4py_arc)
                
            # Add out arc and in arc property to transitions
            if arc.source in [t.name for t in pm4py_pn.transitions]:
                transition = pm4py_dict[arc.source]
                transition.out_arcs.add(pm4py_arc)
            else:
                transition = pm4py_dict[arc.target]
                transition.in_arcs.add(pm4py_arc)

        source = pm4py_dict[self.get_start_place().name]
        target = pm4py_dict[self.get_end_place().name]
        initial_marking = Marking({source: 1})
        final_marking = Marking({target: 1})
        
        
        return pm4py_pn, initial_marking, final_marking

    @classmethod
    def from_pm4py(cls, pm4py_pn):
        """Create a Petri net from a pm4py Petri net.

        Args:
            pm4py_pn (_type_): petri net object from pm4py

        Returns:
            PetriNet: Petri net object
        """
        places = [Place(p.name) for p in pm4py_pn.places]
        index = 0
        transitions = []
        for t in pm4py_pn.transitions:
            if t.label is None:
                t.name = f"tau_{index}"
                index += 1
                transitions.append(Transition(t.name))
            else:
                t.name = t.label
                transitions.append(Transition(t.name))
                
        arcs = []
        for arc in pm4py_pn.arcs:
            source_name = arc.source.name
            target_name = arc.target.name
            weight = arc.weight
            arcs.append(Arc(source_name, target_name, weight))

        # Add token to start place
        converted_pn = cls(places, transitions, arcs)
        start_place = converted_pn.get_start_place()
        start_place.tokens = 1

        return deepcopy(converted_pn)

    @staticmethod
    def from_ptml(ptml_file: str):
        """Create a Petri net from a PTML file."""
        pt = import_ptml_tree(ptml_file)
        pm4py_pn, _, _ = convert_pt_to_pn(pt)
        return deepcopy(PetriNet.from_pm4py(pm4py_pn))

    def soundness_check(self) -> bool:
        """Check if the Petri net is sound, i.e. safeness, proper completion, option to complete and absence of dead parts"""
        pm4py_pn, initial_marking, final_marking = self.to_pm4py()
        return check_soundness(pm4py_pn, initial_marking, final_marking)[0]

    def easy_soundness_check(self) -> bool:
        """Check if the Petri net is easy-sound, i.e. reachability ensured but dead transitions can be present"""
        pm4py_pn, initial_marking, final_marking = self.to_pm4py()
        res = check_easy_soundness_net_in_fin_marking(
            pm4py_pn, initial_marking, final_marking
        )
        return res

    def connectedness_check(self) -> bool:
        """Check if the Petri net is connected, i.e. all transitions must either have an input or an output arc"""
        for t in self.transitions:
            output_arcs = [arc for arc in self.arcs if arc.source == t.name]
            input_arcs = [arc for arc in self.arcs if arc.target == t.name]
            if len(output_arcs) == 0 and len(input_arcs) == 0:
                return False
        return True

    def play_out(self, n: int):
        """Play out the Petri net n times and return the event log.

        Args:
            n (int): number of traces to produce

        Raises:
            ValueError: missing start or end place
            ValueError: deadlock reached

        Returns:
            EventLog: event log object
        """
        if self.get_start_place() is None or self.get_end_place() is None:
            raise ValueError("WF net must have a start and end place to play out")

        def reset_petri_net():
            for place in self.places:
                place.tokens = 0
            start_place = self.get_start_place()
            start_place.tokens = 1

        # Play out the Petri net n times
        event_log = []
        for _ in range(n):
            trace = []
            while self.get_end_place().tokens == 0:
                enabled_transitions = [
                    t for t in self.transitions if self.is_transition_enabled(t.name)
                ]

                # If only one transition is enabled
                if len(enabled_transitions) == 1:
                    self.fire_transition(enabled_transitions[0])
                    trace.append(enabled_transitions[0].name)

                # If multiple transitions are enabled, choose one randomly
                if len(enabled_transitions) > 1:
                    random_transition = random.choice(enabled_transitions)
                    self.fire_transition(random_transition)
                    trace.append(random_transition.name)

                # If deadlock is reached, raise an error
                if len(enabled_transitions) == 0:
                    raise ValueError(
                        "The WF net contains a deadlock and cannot be played out"
                    )

            event_log.append(trace)
            reset_petri_net()

        return EventLog.from_trace_list(event_log)
    
    def is_place(self, graph_in_going_transitions, graph_out_going_transitions):
        """
        Check if the place is a true place, supporting one-to-one, one-to-many, and many-to-many relationships.

        Parameters:
        ----------
        in_going_transitions : list[str]
            List of activity names representing incoming transitions to the place.
        out_going_transitions : list[str]
            List of activity names representing outgoing transitions from the place.

        Returns:
        -------
        bool
            True if every `in_going_transitions` is connected to every `out_going_activity` through the place,
            False otherwise.
        """
        for place in self.places:
            in_going_transitions_for_place = [arc.source for arc in self.arcs if arc.target == place.name]
            out_going_transitions_for_place = [arc.target for arc in self.arcs if arc.source == place.name]
            if set(in_going_transitions_for_place) == set(graph_in_going_transitions) and set(out_going_transitions_for_place) == set(graph_out_going_transitions):
                return True

        return False

    def to_ptml(self, filename: str):
        """Convert the Petri net to a PTML file."""
        pm4py_pn, init, end = self.to_pm4py()
        pt = convert_to_pt(pm4py_pn, init, end)
        write_ptml(pt, filename)
        print(f"Petri net saved as {filename}")
    
    def empty(self):
        """Empty the Petri net."""
        self.places = []
        self.transitions = []
        self.arcs = []
      
    @staticmethod
    def from_graph(graph: Data):
        """
        Populate the Petri net from a PyTorch Geometric graph, considering only selected nodes.

        Parameters:
        -----------
        graph : PyTorch Geometric Data
            The graph containing nodes, edges, node attributes, and selected nodes.
        """
        pn = PetriNet(arcs=[], places=[], transitions=[])

        # Extract the necessary data
        node_names = graph['nodes']
        node_types = graph['node_types']
        edge_index = graph['edge_index']
        selected_nodes = graph['selected_nodes']

        # Create Place and Transition objects only for selected nodes
        node_map = {}  # A map from node index to Place or Transition object

        for i, (name, is_selected) in enumerate(zip(node_names, selected_nodes)):
            if not is_selected:
                continue  # Skip non-selected nodes

            if node_types[i] == 'place':  # Node is a place
                place = Place(name)
                pn.places.append(place)
                node_map[i] = place
            elif node_types[i] == 'transition':  # Node is a transition
                transition = Transition(name)
                pn.transitions.append(transition)
                node_map[i] = transition

        # Create Arc objects for edges that connect selected nodes
        for src, dst in edge_index.t().tolist():
            if src in node_map and dst in node_map:
                source = node_map[src]
                target = node_map[dst]

                # Determine direction of the arc
                if isinstance(source, Place) and isinstance(target, Transition):
                    arc = Arc(source.name, target.name)   
                elif isinstance(source, Transition) and isinstance(target, Place):
                    arc = Arc(source.name, target.name)
                else:
                    # Skip any edges that don't fit the Place-Transition or Transition-Place pattern
                    continue

                pn.arcs.append(arc)
                
        # add a start and end place 
        pn.construct_start_place()
        pn.construct_end_place()        
        return pn