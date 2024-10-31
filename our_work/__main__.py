from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from PyProcTree.exposed_func import generate_logs, generate_process_trees

if __name__ == "__main__":
    # use PyProcTree to generate 
    
    file = "./example/data.xes"
    eventlog = EventLog.load_xes(file)
    
    graph_builder = GraphBuilder()
    petrinet_graph = graph_builder.build_petrinet_graph(eventlog)
    
    print(eventlog)
    for key, value in petrinet_graph.items():
        print(key)
        print(value)
    
    
