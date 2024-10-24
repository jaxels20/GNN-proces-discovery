from EventLog import EventLog
from PetriNet import PetriNet


if __name__ == "__main__":
    # file_name = "data.xes"
    # eventlog = EventLog()
    # eventlog.load_xes(file_name)
    
    # print(eventlog)
    
    # Create a Petri net
    net = PetriNet()

    # Add places with tokens
    net.add_place("P1", tokens=2)
    net.add_place("P2", tokens=0)

    # Add transitions
    net.add_transition("T1")
    #net.add_transition("T2")

    # Add arcs
    net.add_arc(net.places[0], net.transitions[0])  # Correctly passing Place and Transition objects
    net.add_arc(net.transitions[0], net.places[1])  # Correctly passing Transition and Place objects

    # convert it to a pm4py Petri net
    pm4_py_pet = net.to_pm4py()
    
    # convert it back to our Petri net
    net2 = PetriNet.from_pm4py(pm4_py_pet)
    
    visualizer = net2.visualize()
    
        