from src.PetriNet import PetriNet, Transition


def compare_discovered_pn_to_true_pn(candidate_pn: PetriNet, true_pn: PetriNet):
    """
    Compare the discovered Petri net to the true Petri net and return the number of 
    true positives, false positives, and false negatives.
    """
    def normalize_transition_name(name: str):
        return "tau" if "tau" in name else name
    
    true_places = true_pn.places
    candidate_places = candidate_pn.places
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for candidate_place in candidate_places:
        found_true_positive = False
        candidate_ingoing_transitions = sorted([normalize_transition_name(t.name) for t in candidate_pn.get_ingoing_transitions(candidate_place.name)])
        candidate_outgoing_transitions = sorted([normalize_transition_name(t.name) for t in candidate_pn.get_outgoing_transitions(candidate_place.name)])  
        for true_place in true_places:
            true_ingoing_transitions = sorted([normalize_transition_name(t.name) for t in true_pn.get_ingoing_transitions(true_place.name)])
            true_outgoing_transitions = sorted([normalize_transition_name(t.name) for t in true_pn.get_outgoing_transitions(true_place.name)])
            if candidate_ingoing_transitions == true_ingoing_transitions and candidate_outgoing_transitions == true_outgoing_transitions:
                true_positives += 1
                found_true_positive = True
                break
        if not found_true_positive:
            false_positives += 1
                
    # do the same for false negatives
    for true_place in true_places:
        found_false_negative = True
        true_ingoing_transitions = sorted([normalize_transition_name(t.name) for t in true_pn.get_ingoing_transitions(true_place.name)])
        true_outgoing_transitions = sorted([normalize_transition_name(t.name) for t in true_pn.get_outgoing_transitions(true_place.name)])
        for candidate_place in candidate_places:
            candidate_ingoing_transitions = sorted([normalize_transition_name(t.name) for t in candidate_pn.get_ingoing_transitions(candidate_place.name)])
            candidate_outgoing_transitions = sorted([normalize_transition_name(t.name) for t in candidate_pn.get_outgoing_transitions(candidate_place.name)])
            if candidate_ingoing_transitions == true_ingoing_transitions and candidate_outgoing_transitions == true_outgoing_transitions:
                found_false_negative = False
        if found_false_negative:
            false_negatives += 1

    return true_positives, false_positives, false_negatives
  