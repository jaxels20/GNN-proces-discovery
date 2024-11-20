from pm4py.objects.process_tree.importer.variants import ptml
from pm4py.visualization.process_tree import visualizer
from data_generation.data_generation import generate_single_tree, load_parameters
import json
SAVE_PATH = "./data_generation/analysis_data"

if __name__ == "__main__":
    args = {"bg_color": "white", "rankdir": "TB"}
    # Load parameters
    with open("./data_generation/analysis_params.json", "r") as file:
        config = json.load(file)
    
    for tree, params in config.items():
        pt = generate_single_tree(params)
        gviz = visualizer.apply(pt, args)
        visualizer.save(gviz, f"{SAVE_PATH}/{tree}.png")