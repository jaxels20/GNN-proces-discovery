from EventLog import EventLog
from GraphBuilder import GraphBuilder

if __name__ == "__main__":
    file_name = "data.xes"
    eventlog = EventLog.load_xes(file_name)
    
    # Build a Petri net graph
    graph_builder = GraphBuilder()
    petrinet_graph = graph_builder.build_trace_graph(eventlog)
    
    print(petrinet_graph)    
    
    # Visualize the graph
    graph_builder.visualize_graph(petrinet_graph)
    
    
    
    
    
    
    
        