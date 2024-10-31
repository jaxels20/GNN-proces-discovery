from EventLog import EventLog
from GraphBuilder import GraphBuilder
from PetriNet import PetriNet, Place, Transition, Arc
from PyProcTree.exposed_func import generate_logs, generate_process_trees

if __name__ == "__main__":
    file_name = "./example/tree_1_1.ptml"
    pn = PetriNet.from_ptml(file_name)
    pn.visualize("./example/sound_net")
