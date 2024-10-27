from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.analysis import check_soundness, check_is_workflow_net


if __name__ == "__main__":
    # Create simple example of petri and visualize it
    petri_net = PetriNet(
        places=[Place("p1", 1), Place("p2", 0), Place("p3", 0)],
        arcs=[Arc("p1", "t1"), Arc("t1", "p2"), Arc("p2", "t2"), Arc("t2", "p3"), Arc("p1", "t3"), Arc("t3", "p3"), Arc("p2", "t3")],
        transitions=[Transition("t1"), Transition("t2"), Transition("t3")]
    )
    
    # petri_net.visualize("./example/petri_net")
    pm4py_net, init, end = petri_net.to_pm4py()
    
    # visualize the pm4py petri net
    gviz = pn_visualizer.apply(pm4py_net, init, end)
    pn_visualizer.view(gviz)
    
    # print(petri_net.soundness_check())
    # print(petri_net.easy_soundness_check())
    # print(petri_net.connectedness_check())