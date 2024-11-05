from PetriNet import PetriNet, Place, Transition, Arc
import random


if __name__ == "__main__":
    places = [Place('start', tokens=1), Place('end'), Place('ABG->CD')]
    transitions = [Transition('A'), Transition('B'), Transition('C'), Transition('D'), Transition('G')]
    archs = [
        Arc('start', 'A'), 
        Arc('start', 'B'),
        Arc('start', 'G'),
        Arc('G', 'ABG->CD'), 
        Arc('A', 'ABG->CD'), 
        Arc('B', 'ABG->CD'), 
        Arc('ABG->CD', 'C'), 
        Arc('ABG->CD', 'D'), 
        Arc('D', 'end'), 
        Arc('C', 'end')
    ]
    
    pn = PetriNet(places, transitions, archs)
    
    pn.to_ptml('./example/many_to_many.ptml')
    