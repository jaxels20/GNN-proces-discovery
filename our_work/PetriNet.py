# from graphviz import Digraph
# from pm4py.objects.petri_net.utils import petri_utils
# from pm4py.objects.petri_net.obj import PetriNet as PM4PyPetriNet, Marking

# class Place:
#     """
#     Class representing a place in a Petri net.
    
#     Attributes:
#     -----------
#     name : str
#         The name of the place.
#     tokens : int
#         The number of tokens in the place.
#     """
#     def __init__(self, name: str, tokens: int = 0):
#         self.name = name
#         self.tokens = tokens

#     def add_tokens(self, count: int = 1):
#         """Add tokens to the place."""
#         self.tokens += count

#     def remove_tokens(self, count: int = 1):
#         """Remove tokens from the place, ensuring no negative token count."""
#         if self.tokens - count < 0:
#             raise ValueError(f"Cannot remove {count} tokens from place '{self.name}' (tokens = {self.tokens})")
#         self.tokens -= count

# class Transition:
#     """
#     Class representing a transition in a Petri net.
    
#     Attributes:
#     -----------
#     name : str
#         The name of the transition.
#     """
#     def __init__(self, name: str):
#         self.name = name

# class Arc:
#     """
#     Class representing an arc in a Petri net, connecting a place and a transition.
    
#     Attributes:
#     -----------
#     source : Place or Transition
#         The source of the arc (either a Place or a Transition).
#     target : Place or Transition
#         The target of the arc (either a Place or a Transition).
#     weight : int
#         The weight of the arc.
#     """
#     def __init__(self, source, target, weight: int = 1):
#         self.source = source
#         self.target = target
#         self.weight = weight

# class PetriNet:
#     """
#     Class representing a generic Petri net.

#     Attributes:
#     -----------
#     places : list[Place]
#         List of places in the Petri net.
#     transitions : list[Transition]
#         List of transitions in the Petri net.
#     arcs : list[Arc]
#         List of arcs connecting places and transitions.
#     """
#     def __init__(self):
#         self.places = []
#         self.transitions = []
#         self.arcs = []

#     def add_place(self, name: str, tokens: int = 0):
#         """Add a place to the Petri net."""
#         place = Place(name, tokens)
#         self.places.append(place)
    
#     def add_transition(self, name: str):
#         """Add a transition to the Petri net."""
#         transition = Transition(name)
#         self.transitions.append(transition)

#     def add_arc(self, source, target, weight: int = 1):
#         """Add an arc connecting a place and a transition or vice versa."""
#         arc = Arc(source, target, weight)
#         self.arcs.append(arc)

#     def get_place_by_name(self, name: str):
#         """Return a place by its name, or None if it doesn't exist."""
#         for place in self.places:
#             if place.name == name:
#                 return place
#         return None

#     def get_transition_by_name(self, name: str):
#         """Return a transition by its name, or None if it doesn't exist."""
#         for transition in self.transitions:
#             if transition.name == name:
#                 return transition
#         return None

#     def is_transition_enabled(self, transition: Transition) -> bool:
#         """
#         Check if a transition is enabled.
        
#         A transition is enabled if all its input places have enough tokens.
#         """
#         input_arcs = [arc for arc in self.arcs if arc.target == transition]
#         return all(arc.source.tokens >= arc.weight for arc in input_arcs)

#     def fire_transition(self, transition: Transition):
#         """
#         Fire a transition if it is enabled, moving tokens from input places to output places.
        
#         Raises:
#         -------
#         ValueError : If the transition is not enabled.
#         """
#         if not self.is_transition_enabled(transition):
#             raise ValueError(f"Transition '{transition.name}' is not enabled")

#         # Remove tokens from input places
#         input_arcs = [arc for arc in self.arcs if arc.target == transition]
#         for arc in input_arcs:
#             arc.source.remove_tokens(arc.weight)

#         # Add tokens to output places
#         output_arcs = [arc for arc in self.arcs if arc.source == transition]
#         for arc in output_arcs:
#             arc.target.add_tokens(arc.weight)

#     def __repr__(self):
#         return f"PetriNet(Places: {len(self.places)}, Transitions: {len(self.transitions)}, Arcs: {len(self.arcs)})"

#     def visualize(self, filename="petri_net", format="png"):
#         """
#         Visualize the Petri net and save it as a PNG file using Graphviz.

#         Parameters:
#         -----------
#         filename : str
#             The base filename for the output file (without extension).
#         format : str
#             The format for the output file (e.g., 'png', 'pdf').
#         """
#         dot = Digraph(comment="Petri Net", format=format)

#         # Add places as circles, displaying the number of tokens
#         for place in self.places:
#             label = f"{place.name}\nTokens: {place.tokens}"
#             dot.node(place.name, label=label, shape="circle", color="lightblue", style="filled")

#         # Add transitions as rectangles
#         for transition in self.transitions:
#             dot.node(transition.name, label=transition.name, shape="box", color="lightgreen", style="filled")

#         # Add arcs (directed edges between places and transitions)
#         for arc in self.arcs:
#             dot.edge(arc.source.name, arc.target.name, label=str(arc.weight))

#         # Save the Petri net as a file
#         output_path = dot.render(filename)
#         print(f"Petri net saved as {output_path}")

#     def get_start_place(self):
#         """Return the start place (no incoming arcs), or None if none exists."""
#         for place in self.places:
#             incoming_arcs = [arc for arc in self.arcs if arc.target == place]
#             if len(incoming_arcs) == 0:
#                 return place
#         return None

#     def get_end_place(self):
#         """Return the end place (no outgoing arcs), or None if none exists."""
#         for place in self.places:
#             outgoing_arcs = [arc for arc in self.arcs if arc.source == place]
#             if len(outgoing_arcs) == 0:
#                 return place
#         return None

#     def to_pm4py(self):
#         # Step 1: Create an empty pm4py Petri net
#         pm4py_pn = PM4PyPetriNet("Custom_Petri_Net")

#         # Step 2: Create places
#         place_dict = {}
#         for place_name in self.places:
#             place = PM4PyPetriNet.Place(place_name)
#             pm4py_pn.places.add(place)
#             place_dict[place_name] = place

#         # Step 3: Create transitions
#         transition_dict = {}
#         for transition_name, silent in self.transitions:
#             if silent:
#                 transition = PM4PyPetriNet.Transition(transition_name, None)  # Silent transition
#             else:
#                 transition = PM4PyPetriNet.Transition(transition_name, transition_name)
#             pm4py_pn.transitions.add(transition)
#             transition_dict[transition_name] = transition

#         # Step 4: Create arcs
#         for arc in self.arcs:
#             source_name, target_name = arc
#             source = place_dict.get(source_name) or transition_dict.get(source_name)
#             target = place_dict.get(target_name) or transition_dict.get(target_name)
#             if source and target:
#                 petri_utils.add_arc(pm4py_pn, source, target)

#         # Step 5: Define initial and final markings (optional)
#         initial_marking = Marking()
#         final_marking = Marking()
#         for place in self.initial_marking:
#             initial_marking[place_dict[place]] = 1
#         for place in self.final_marking:
#             final_marking[place_dict[place]] = 1

#         return pm4py_pn, initial_marking, final_marking

#     @classmethod
#     def from_pm4py(cls, pm4py_pn, initial_marking=None, final_marking=None):
#         # Step 1: Extract places
#         places = [p.name for p in pm4py_pn.places]

#         # Step 2: Extract transitions
#         transitions = [(t.name, t.label is None) for t in pm4py_pn.transitions]  # True for silent transitions

#         # Step 3: Extract arcs
#         arcs = []
#         for arc in pm4py_pn.arcs:
#             source_name = arc.source.name
#             target_name = arc.target.name
#             arcs.append((source_name, target_name))

#         # Step 4: Extract markings
#         initial_marking_list = [p.name for p in initial_marking] if initial_marking else []
#         final_marking_list = [p.name for p in final_marking] if final_marking else []

#         # Step 5: Return a new instance of the custom Petri net
#         return cls(places, transitions, arcs, initial_marking_list, final_marking_list)
