import constants as const
import matplotlib.pyplot as plt
from pm4py.objects.process_tree.obj import ProcessTree, Operator

from pm4py.visualization.process_tree.visualizer import apply as pt_visualizer
from pm4py.visualization.petri_net.visualizer import apply as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter

import sys 
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from src.PetriNet import PetriNet


def process_tree_fig():
    # Step 1: Define activity nodes
    activity_a = ProcessTree(label="a")  # Activity node labeled "A"
    activity_b = ProcessTree(label="b")  # Activity node labeled "B"
    activity_c = ProcessTree(label="c")  # Activity node labeled "C"
    activity_d = ProcessTree(label="d")  # Activity node labeled "D"


    # Step 2: Create the root node with SEQUENCE operator (➡️ symbol)
    root_node = ProcessTree(operator=Operator.SEQUENCE, label="→")

    # Step 3: Create additional nodes with operator symbols
    xor_node = ProcessTree(operator=Operator.XOR, label="⊕")  # XOR operator symbol
    parallel_node = ProcessTree(operator=Operator.PARALLEL, label="∥")  # Parallel operator symbol
    loop_node = ProcessTree(operator=Operator.LOOP, label="⟳")  # Loop operator symbol


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
    
    # Step 7: Explicitly set labels for operator nodes in the Graphviz object
    for node_id in gviz.node_attr.keys():
        if node_id.startswith("operator:"):
            operator_type = gviz.node_attr[node_id].get('label', '').strip('"')
            # Map operator labels to custom symbols
            symbol_map = {
                "sequence": "→",
                "xor": "⊕",
                "and": "∥",
                "loop": "⟳",
            }
            if operator_type.lower() in symbol_map:
                gviz.node_attr[node_id]['label'] = f'"{symbol_map[operator_type.lower()]}"'

    
    # set the rankdir to be from left to right
    gviz.attr(
        rankdir='TB',
        size= f'{const.FIG_WIDTH},{const.FIG_HEIGHT}!',
              )
    

    # Save the modified Graphviz object as a PDF file
    gviz.render(filename=const.OUTPUT_DIR + 'process_tree', format='pdf', cleanup=True)

    return root_node

def petri_net_fig(tree: ProcessTree):
    # Step 1: Convert the Process Tree to a Petri Net
    net, initial_marking, final_marking = pt_converter.apply(tree)
    our_net = PetriNet.from_pm4py(net)
    
    # Step 2: Visualize the Petri Net
    gviz = our_net.get_visualization("pdf", True)
    
    gviz.attr(
        rankdir='LR',
        size= f'{const.FIG_WIDTH},{const.FIG_HEIGHT}!',
        dpi=str(const.DPI),
        fontname=const.FONT_FAMILY,
        fontsize=str(const.FONT_SIZE),
        bgcolor='white',  # Or another color to match `plt.style`
        margin='0,0,0,0'
        )
    
    # Step 3: Save the Petri Net as a PDF
    gviz.render(filename=const.OUTPUT_DIR + 'petri_net', format='pdf', cleanup=True)

if __name__ == '__main__':
    # Process Tree Figure
    tree = process_tree_fig()
    
    # Petri Net Figure
    petri_net_fig(tree)
    

    


    
