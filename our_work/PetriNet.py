from graphviz import Digraph
from pm4py.objects.petri_net.obj import PetriNet as PM4PyPetriNet, Marking
from pm4py.analysis import check_soundness
from pm4py.objects.petri_net.utils.check_soundness import check_easy_soundness_net_in_fin_marking


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
            raise ValueError(f"Cannot remove {count} tokens from place '{self.name}' (tokens = {self.tokens})")
        self.tokens -= count

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
    def __init__(self, source, target, weight: int = 1):
        self.source = source
        self.target = target
        self.weight = weight

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

    def add_arc(self, source, target, weight: int = 1):
        """Add an arc connecting a place and a transition or vice versa."""
        if source not in [place.name for place in self.places] + [transition.name for transition in self.transitions]:
            raise ValueError(f"Source '{source}' does not exist in the Petri net")
        if target not in [place.name for place in self.places] + [transition.name for transition in self.transitions]:
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

    def is_transition_enabled(self, transition: Transition) -> bool:
        """
        Check if a transition is enabled.
        
        A transition is enabled if all its input places have enough tokens.
        """
        input_arcs = [arc for arc in self.arcs if arc.target == transition]
        return all(arc.source.tokens >= arc.weight for arc in input_arcs)

    def fire_transition(self, transition: Transition):
        """
        Fire a transition if it is enabled, moving tokens from input places to output places.
        
        Raises:
        -------
        ValueError : If the transition is not enabled.
        """
        if not self.is_transition_enabled(transition):
            raise ValueError(f"Transition '{transition.name}' is not enabled")

        # Remove tokens from input places
        input_arcs = [arc for arc in self.arcs if arc.target == transition]
        for arc in input_arcs:
            arc.source.remove_tokens(arc.weight)

        # Add tokens to output places
        output_arcs = [arc for arc in self.arcs if arc.source == transition]
        for arc in output_arcs:
            arc.target.add_tokens(arc.weight)

    def __repr__(self):
        return f"PetriNet(Places: {len(self.places)}, Transitions: {len(self.transitions)}, Arcs: {len(self.arcs)})"

    def visualize(self, filename="petri_net", format="png"):
        """
        Visualize the Petri net and save it as a PNG file using Graphviz.

        Parameters:
        -----------
        filename : str
            The base filename for the output file (without extension).
        format : str
            The format for the output file (e.g., 'png', 'pdf').
        """
        dot = Digraph(comment="Petri Net", format=format)

        for place in self.places:
            label = f"{place.name}\nTokens: {place.tokens}"
            dot.node(place.name, label=label, shape="circle", color="lightblue", style="filled")

        for transition in self.transitions:
            dot.node(transition.name, label=transition.name, shape="box", color="lightgreen", style="filled")

        for arc in self.arcs:
            dot.edge(arc.source, arc.target, label=str(arc.weight))

        output_path = dot.render(filename, cleanup=True)
        print(f"Petri net saved as {output_path}")

    def get_start_place(self):
        """Return the start place (no incoming arcs), or None if none exists."""
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

    def to_pm4py(self):
        """Convert our Petri net class to a pm4py Petri net."""
        pm4py_pn = PM4PyPetriNet("Custom_Petri_Net")
        pm4py_dict = {}

        for place in self.places:
            pm4py_place = PM4PyPetriNet.Place(place.name)
            pm4py_pn.places.add(pm4py_place)
            pm4py_dict[place.name] = pm4py_place
            
        for transition in self.transitions:
            pm4py_transition = PM4PyPetriNet.Transition(transition.name, transition.name)
            pm4py_pn.transitions.add(pm4py_transition)
            pm4py_dict[transition.name] = pm4py_transition
            
        for arc in self.arcs:
            pm4py_arc = PM4PyPetriNet.Arc(pm4py_dict[arc.source], pm4py_dict[arc.target])
            pm4py_pn.arcs.add(pm4py_arc)

        initial_marking = Marking()
        initial_marking[pm4py_dict[self.get_start_place().name]] = 1
        final_marking = Marking()
        final_marking[pm4py_dict[self.get_end_place().name]] = 1

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
        transitions = [Transition(t.name) for t in pm4py_pn.transitions]
        arcs = []
        for arc in pm4py_pn.arcs:
            source_name = arc.source.name
            target_name = arc.target.name
            weight = arc.weight
            arcs.append(Arc(source_name, target_name, weight))

        return cls(places, transitions, arcs)

    def soundness_check(self) -> bool:
        """Check if the Petri net is sound"""
        pm4py_pn, initial_marking, final_marking = self.to_pm4py()
        return check_soundness(pm4py_pn, initial_marking, final_marking)
    
    def easy_soundness_check(self) -> bool:
        """Check if the Petri net is sound"""
        pm4py_pn, initial_marking, final_marking = self.to_pm4py()
        return check_easy_soundness_net_in_fin_marking(pm4py_pn, initial_marking, final_marking)
    
    def connectedness_check(self) -> bool:
        """Check if the Petri net is connected, i.e. all transitions must either have an input or an output arc"""
        for t in self.transitions:
            output_arcs = [arc for arc in self.arcs if arc.source == t.name]
            input_arcs = [arc for arc in self.arcs if arc.target == t.name]
            if len(output_arcs) == 0 and len(input_arcs) == 0:
                return False
        return True