from EventLog import EventLog
from GraphBuilder import GraphBuilder

if __name__ == "__main__":
    file_name = "data.xes"
    eventlog = EventLog.load_xes(file_name)
    print(eventlog)
    
    foot_print = eventlog.get_footprint_matrix()
    
    # Build a Petri net graph
    graph_builder = GraphBuilder()
    petrinet_graph = graph_builder.build_petrinet_graph(eventlog)
    
    # print the content of the graph
    for key, value in petrinet_graph.items():
        print(key, value)
    
    
    
    
    
    
    
    
        