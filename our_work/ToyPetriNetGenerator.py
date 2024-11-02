from PetriNet import PetriNet, Place, Transition, Arc
import random


if __name__ == "__main__":
    places = [Place('start', tokens=1), Place('end'), Place('A->BDE'), Place('BDE->C') ]
    
    transitions = [Transition('A'), Transition('B'), Transition('C'), Transition('D'), Transition('E')]
    
    archs = [Arc('start', 'A'), Arc('A', 'A->BDE'), Arc('A->BDE', 'B'), Arc('A->BDE', 'D'), Arc('A->BDE', 'E'), 
             Arc('B', 'BDE->C'), Arc('D', 'BDE->C'), Arc('E', 'BDE->C'), Arc('BDE->C', 'C'), Arc('C', 'end')]
    
    pn = PetriNet(places, transitions, archs)
    
    pn.to_ptml('example/toy_pn.ptml')
    