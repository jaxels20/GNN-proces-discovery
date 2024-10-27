from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from pm4py.visualization.petri_net import visualizer as pn_visualizer


if __name__ == "__main__":
    # SOUND NET EXAMPLE
    test_2_p = [Place("p1", 1), Place("p2", 0), Place("p3", 0)]
    test_2_t = [Transition("t1"), Transition("t2")]
    test_2_a = [Arc("p1", "t1"), Arc("t1", "p2"), Arc("p2", "t2"), Arc("t2", "p3")]
    # EASY SOUND EXAMPLE
    test_1_p = [Place("p1", 1), Place("p2", 0), Place("p3", 0)]
    test_1_t = [Transition("t1"), Transition("t2"), Transition("t3")]
    test_1_a = [Arc("p1", "t1"), Arc("t1", "p2"), Arc("p2", "t2"), Arc("t2", "p3"), Arc("p1", "t3"), Arc("t3", "p3"), Arc("p2", "t3")]
    
    petri_net_1 = PetriNet(
        places=test_1_p,
        arcs=test_1_a,
        transitions=test_1_t
    )
    # petri_net_1.visualize("./example/petri_net_1")
    petri_net_2 = PetriNet(
        places=test_2_p,
        arcs=test_2_a,
        transitions=test_2_t
    )
    # petri_net_2.visualize("./example/petri_net_2")
    # Problem with easy sound check always returning false
    print(f"Sound net example - Soundness check: {petri_net_2.soundness_check()}")
    print(f"Sound net example - Easy sound check: {petri_net_2.easy_soundness_check()}")
    print(f"Sound net example - Connectedness: {petri_net_2.connectedness_check()}")
    print(f"Easy-sound net example - Soundness check: {petri_net_1.soundness_check()}")
    print(f"Easy-sound net example - Easy sound check: {petri_net_1.easy_soundness_check()}")
    print(f"Easy-sound net example - Connectedness: {petri_net_1.connectedness_check()}")