from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from PyProcTree.exposed_func import generate_logs, generate_process_trees

if __name__ == "__main__":
    file_name = "./example/data.xes"
    
    eventlog = EventLog.load_xes(file_name)
    
    places = [Place('start', tokens=1), Place('end'), Place('A->BDE'), Place('BDE->C') ]
    transitions = [Transition('A'), Transition('B'), Transition('C'), Transition('D'), Transition('E')]
    archs = [Arc('start', 'A'), Arc('A', 'A->BDE'), Arc('A->BDE', 'B'), Arc('A->BDE', 'D'), Arc('A->BDE', 'E'), 
             Arc('B', 'BDE->C'), Arc('D', 'BDE->C'), Arc('E', 'BDE->C'), Arc('BDE->C', 'C'), Arc('C', 'end')]
    pn = PetriNet(places, transitions, archs)
    

    gb = GraphBuilder()
    graph = gb.build_petrinet_graph(eventlog)
    
    graph = gb.annotate_petrinet_graph(graph, pn)
    
    for key, value in graph.items():
        print(key, value)
    
    