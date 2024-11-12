"""
This script will generate a set of controlled scenario cases for the process mining experiments. it wil out in the 
out folder a ptml file and a xes file. the are coupled by the same name. Usage:
python generate_controlled_scenario_cases.py --output_dir "
"""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from EventLog import EventLog
from ToyDataGenerator import ToyDataGenerator
from PetriNet import PetriNet, Transition, Place
# append parent directory to path


# TEST CASE 1: A simple sequence of activities
def simple_sequence(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["ABC", "ABC"]
    sub_folder_name = "simple_sequence/"
    
    #Check if the folder exists
    os.makedirs(f"{output_dir}{sub_folder_name}", exist_ok=True)
    
    
    ToyDataGenerator.traces_to_xes(traces, f"{output_dir}{sub_folder_name}eventlog.xes")
    
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()

    
    # Add places and transitions to the Petri net
    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    petri_net.add_place("A->B")
    petri_net.add_place("B->C")
    
    
    # Add arcs to the Petri net
    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->B")
    petri_net.add_arc("A->B", "B")
    petri_net.add_arc("B", "B->C")
    petri_net.add_arc("B->C", "C")
    petri_net.add_arc("C", "End")
    
    petri_net.visualize(f"{output_dir}{sub_folder_name}petri_net")
    petri_net.to_ptml(f"{output_dir}{sub_folder_name}petri_net.ptml")


# TEST CASE 2: A simple XOR split
def simple_xor_split(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["AB", "AC"]
    subfolder_name = "simple_xor_split"
    
    #Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)
    
    ToyDataGenerator.traces_to_xes(traces, f"{output_dir}{subfolder_name}/eventlog.xes")
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()
    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    
    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    petri_net.add_place("A->C,D")

    
    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->C,D")
    petri_net.add_arc("A->C,D", "C")
    petri_net.add_arc("A->C,D", "B")
    petri_net.add_arc("B", "End")
    petri_net.add_arc("C", "End")

    
    
    petri_net.visualize(f"{output_dir}{subfolder_name}/petri_net")
    petri_net.to_ptml(f"{output_dir}{subfolder_name}/petri_net.ptml")
    
# TEST CASE 3: A simple AND split
def simple_and_split(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["ABCD", "ACBD"]
    subfolder_name = "simple_and_split"
    
    #Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)
    
    ToyDataGenerator.traces_to_xes(traces, f"{output_dir}{subfolder_name}/eventlog.xes")
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()

    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    petri_net.add_transition("D")
    
    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    
    petri_net.add_place("A->B")
    petri_net.add_place("A->C")
    petri_net.add_place("B->D")
    petri_net.add_place("C->D")
    
    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->B")
    petri_net.add_arc("A", "A->C")
    petri_net.add_arc("A->B", "B")
    petri_net.add_arc("A->C", "C")
    petri_net.add_arc("B", "B->D")
    petri_net.add_arc("C", "C->D")
    petri_net.add_arc("B->D", "D")
    petri_net.add_arc("C->D", "D")
    petri_net.add_arc("D", "End")
    
    
    petri_net.visualize(f"{output_dir}{subfolder_name}/petri_net")
    petri_net.to_ptml(f"{output_dir}{subfolder_name}/petri_net.ptml")

# TEST CASE 4: A simple loop of lenght 1


# TEST CASE 5: A simple loop of lenght 2

# TEST CASE 6: A long term dependency
def long_dependency(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["ACD", "BCE", "ACD", "BCE"]
    subfolder_name = "long_dependency"
    
    #Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)
    
    ToyDataGenerator.traces_to_xes(traces, f"{output_dir}{subfolder_name}/eventlog.xes")
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()
    
    # Add places and transitions to the Petri net
    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    petri_net.add_transition("D")
    petri_net.add_transition("E")

    
    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    
    petri_net.add_place("A,B->C")
    petri_net.add_place("C->D,E")
    petri_net.add_place("A->D")
    petri_net.add_place("B->E")
     
    
    # Add arcs to the Petri net
    
    petri_net.add_arc("Start", "A")
    petri_net.add_arc("Start", "B")
    petri_net.add_arc("A", "A,B->C")
    petri_net.add_arc("B", "A,B->C")
    petri_net.add_arc("A,B->C", "C")
    petri_net.add_arc("C", "C->D,E")
    petri_net.add_arc("C->D,E", "D")
    petri_net.add_arc("C->D,E", "E")
    petri_net.add_arc("A", "A->D")
    petri_net.add_arc("B", "B->E")
    petri_net.add_arc("A->D", "D")
    petri_net.add_arc("B->E", "E")
    
    petri_net.add_arc("D", "End")
    petri_net.add_arc("E", "End")
    
    petri_net.visualize(f"{output_dir}{subfolder_name}/petri_net")
    petri_net.to_ptml(f"{output_dir}{subfolder_name}/petri_net.ptml")
    
    



if __name__ == "__main__":
    output_dir = "./controlled_scenarios/"
    simple_sequence(output_dir)
    simple_xor_split(output_dir)
    simple_and_split(output_dir)
    long_dependency(output_dir)
    
    