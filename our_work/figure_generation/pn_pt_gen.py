import constants as const
import matplotlib.pyplot as plt
from pm4py.objects.process_tree.obj import ProcessTree, Operator

from pm4py.visualization.process_tree.visualizer import apply as pt_visualizer
from pm4py.visualization.petri_net.visualizer import apply as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter

import sys 


def process_tree_fig():
    # Step 1: Define activity nodes
    activity_a = ProcessTree(label="A")  # Activity node labeled "A"
    activity_b = ProcessTree(label="B")  # Activity node labeled "B"
    activity_c = ProcessTree(label="C")  # Activity node labeled "C"
    activity_d = ProcessTree(label="D")  # Activity node labeled "D"
    activity_e = ProcessTree(label="E")  # Activity node labeled "E"
    activity_f = ProcessTree(label="F")  # Activity node labeled "F"

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
    

    # Step 5: Visualize the process tree as a PDF
    gviz = pt_visualizer(root_node)
    
    # set the rankdir to be from left to right
    gviz.attr(
        rankdir='TB',
        size= f'{const.DOUBLE_COL_FIG_WIDTH},{const.DOUBLE_COL_FIG_HEIGHT}!',
              )
    

    # Save the modified Graphviz object as a PDF file
    gviz.render(filename=const.OUTPUT_DIR + 'process_tree', format='pdf', cleanup=True)

    return root_node

def petri_net_fig(tree: ProcessTree):
    # Step 1: Convert the Process Tree to a Petri Net
    net, initial_marking, final_marking = pt_converter.apply(tree)
    
    
    # Step 2: Visualize the Petri Net
    gviz = pn_visualizer(net, initial_marking, final_marking)
    
    gviz.attr(
        rankdir='TB',
        size= f'{const.SINGLE_COL_FIG_WIDTH},{const.SINGLE_COL_FIG_HEIGHT}!',
              )
    
    # Step 3: Save the Petri Net as a PDF
    gviz.render(filename=const.OUTPUT_DIR + 'petri_net', format='pdf', cleanup=True)

if __name__ == '__main__':
    # Process Tree Figure
    tree = process_tree_fig()
    
    # Petri Net Figure
    petri_net_fig(tree)
    

    


    
