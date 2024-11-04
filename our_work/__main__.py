from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from PyProcTree.exposed_func import generate_logs, generate_process_trees

if __name__ == "__main__":
    file_name = "./example/many_to_many.xes"
    eventlog = EventLog.load_xes(file_name)
    
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
    # pn.visualize("./example/many_to_many")
    # pn = PetriNet.from_ptml('example/toy_pn.ptml')

    gb = GraphBuilder()
    graph = gb.build_petrinet_graph(eventlog)
    graph = gb.annotate_petrinet_graph(graph, pn)
    
    for key, value in graph.items():
        print(key, value)
        print("--------------------------------------------------------")    
    