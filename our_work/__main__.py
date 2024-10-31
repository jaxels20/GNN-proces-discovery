from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc


if __name__ == "__main__":
    file_name = "./example/tree_1_1.ptml"
    pn = PetriNet.from_ptml(file_name)
    pn.visualize("./example/generated_net")