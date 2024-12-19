"""
This script will generate a set of controlled scenario cases for the process mining experiments. it wil out in the 
out folder a ptml file and a xes file. the are coupled by the same name. Usage:
python generate_controlled_scenario_cases.py --output_dir "
"""

import sys
import os
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.process_tree.obj import ProcessTree, Operator

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.EventLog import EventLog
from generate_xes_from_tracelist import TraceListExporter
from src.PetriNet import PetriNet, Transition, Place


# TEST CASE 1: A simple sequence of activities
def simple_sequence(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["ABC", "ABC"]
    sub_folder_name = "simple_sequence/"

    # Check if the folder exists
    os.makedirs(f"{output_dir}{sub_folder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{sub_folder_name}eventlog.xes"
    )

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

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/eventlog.xes"
    )
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

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/eventlog.xes"
    )
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
def loop_lenght_1(output_dir: str) -> None:

    traces = ["ABBC", "ABBBC", "ABBBBC"]
    subfolder_name = "loop_lenght_1"

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/eventlog.xes"
    )
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()

    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")

    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    petri_net.add_place("A->B,C")

    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->B,C")
    petri_net.add_arc("A->B,C", "B")
    petri_net.add_arc("B", "A->B,C")
    petri_net.add_arc("A->B,C", "C")
    petri_net.add_arc("C", "End")

    petri_net.visualize(f"{output_dir}{subfolder_name}/petri_net")
    petri_net.to_ptml(f"{output_dir}{subfolder_name}/petri_net.ptml")


# TEST CASE 5: A simple loop of lenght 2
def loop_lenght_2(output_dir: str) -> None:

    traces = ["ABCD", "ABCBCD", "ABCBCBCD"]
    subfolder_name = "loop_lenght_2"

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/eventlog.xes"
    )
    # Initialize a new Petri net
    petri_net = PetriNet()
    petri_net.empty()

    petri_net.add_transition("A")
    petri_net.add_transition("B")
    petri_net.add_transition("C")
    petri_net.add_transition("D")

    petri_net.add_place("Start", tokens=1)
    petri_net.add_place("End")
    petri_net.add_place("A->B,D")
    petri_net.add_place("B->C")

    petri_net.add_arc("Start", "A")
    petri_net.add_arc("A", "A->B,D")
    petri_net.add_arc("A->B,D", "B")
    petri_net.add_arc("B", "B->C")
    petri_net.add_arc("B->C", "C")
    petri_net.add_arc("C", "A->B,D")
    petri_net.add_arc("A->B,D", "D")
    petri_net.add_arc("D", "End")

    petri_net.visualize(f"{output_dir}{subfolder_name}/petri_net")
    petri_net.to_ptml(f"{output_dir}{subfolder_name}/petri_net.ptml")


# TEST CASE 6: A long term dependency
def long_dependency(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["ACD", "BCE", "ACD", "BCE"]
    subfolder_name = "long_dependency"

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)

    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/eventlog.xes"
    )
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
    petri_net.to_pnml(f"{output_dir}{subfolder_name}/petri_net.pnml")

# TEST CASE 7: Realistic example
def realistic_example(output_dir: str) -> None:
    # Write Petrinet and XES file
    traces = ["AB", "ACDD", "ADDC"]
    subfolder_name = "overleaf_example"

    # Check if the folder exists
    os.makedirs(f"{output_dir}{subfolder_name}", exist_ok=True)
    TraceListExporter.traces_to_xes(
        traces, f"{output_dir}{subfolder_name}/overleaf_example.xes"
    )

    activity_a = ProcessTree(label="a")  # Activity node labeled "A"
    activity_b = ProcessTree(label="b")  # Activity node labeled "B"
    activity_c = ProcessTree(label="c")  # Activity node labeled "C"
    activity_d = ProcessTree(label="d")  # Activity node labeled "D"

    # Step 2: Create the root node with SEQUENCE operator
    root_node = ProcessTree(operator=Operator.SEQUENCE)

    # Step 3: Create additional nodes with operators and assign relationships
    # XOR operator node to decide between B and the parallel section
    xor_node = ProcessTree(operator=Operator.XOR)
    parallel_node = ProcessTree(operator=Operator.PARALLEL)
    loop_node = ProcessTree(operator=Operator.LOOP)

    # Step 4: Build the tree structure by appending children to appropriate parents
    root_node.children = [activity_a, xor_node]  # SEQUENCE of A followed by XOR
    xor_node.children = [activity_b, parallel_node]  # XOR choice: B or PARALLEL block
    parallel_node.children = [activity_c, loop_node]  # PARALLEL execution of C and LOOP
    loop_node.children = [activity_d]  # LOOP around D

    # Step 5: Append the parent node to each child node
    activity_a.parent = root_node
    activity_b.parent = xor_node
    activity_c.parent = parallel_node
    activity_d.parent = loop_node
    xor_node.parent = root_node
    parallel_node.parent = xor_node
    loop_node.parent = parallel_node

    # Step 6: convert to petri net
    net, initial_marking, final_marking = pt_converter.apply(root_node)
    our_net = PetriNet.from_pm4py(net)
    our_net.visualize(f"{output_dir}overleaf_example/overleaf_example")
    our_net.to_ptml(f"{output_dir}overleaf_example/overleaf_example.ptml")


if __name__ == "__main__":
    output_dir = "./controlled_scenarios/"
    simple_sequence(output_dir)
    simple_xor_split(output_dir)
    simple_and_split(output_dir)
    loop_lenght_1(output_dir)
    loop_lenght_2(output_dir)
    long_dependency(output_dir)
    realistic_example(output_dir)
