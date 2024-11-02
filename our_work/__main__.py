from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from PyProcTree.exposed_func import generate_logs, generate_process_trees

if __name__ == "__main__":
    file_name = "./example/data.xes"
    
    eventlog = EventLog.load_xes(file_name)
    
    pn = PetriNet.from_ptml('example/toy_pn.ptml')

    gb = GraphBuilder()
    graph = gb.build_petrinet_graph(eventlog)
    
    graph = gb.annotate_petrinet_graph(graph, pn)
    
    for key, value in graph.items():
        print(key, value)
    
    