from pm4py.visualization.process_tree import visualizer
from pm4py.objects.conversion.process_tree.variants import to_petri_net as pt_to_pn
from pm4py.objects.process_tree.importer.variants import ptml as ptml_importer
from data_generation.data_generation import generate_single_tree
from src.PetriNet import PetriNet
import json
import os


class TrainingDataVisualizer:
    def __init__(self, save_path, pt_viz_args: dict = None):
        self.pt_viz_args = pt_viz_args or {}
        self.save_path = save_path

    @staticmethod
    def _load_config(file_path):
        """Loads configuration from a JSON file."""
        with open(file_path, "r") as file:
            return json.load(file)

    def _save_petri_net(self, pt, tree_name):
        """Converts a process tree to a Petri net and saves its visualization."""
        pn, _, _ = pt_to_pn.apply(pt)
        our_pn = PetriNet.from_pm4py(pn)
        file_path = f"./data_generation/visualizations/pn_{tree_name}"
        our_pn.visualize(file_path)

    def _save_process_tree(self, pt, tree_name):
        """Generates and saves a visualization of the process tree."""
        gviz = visualizer.apply(pt, self.pt_viz_args)
        file_path = f"./data_generation/visualizations/{tree_name}.png"
        visualizer.save(gviz, file_path)
        print(f"Process tree visualization saved to {file_path}")

    def generate_pt_from_config_and_visualize(self, config):
        """Generate process trees based on config and visualize them."""
        for tree_name, params in config.items():
            # Generate process tree
            pt = generate_single_tree(params)

            # Save visualizations
            self._save_process_tree(pt, tree_name)
            self._save_petri_net(pt, tree_name)

    def load_ptml_and_visualize(self, ptml_path):
        """
        Loads a PTML file, visualizes the process tree and Petri net.

        Args:
            ptml_path (str): Path to the PTML file.
        """
        # Ensure the PTML file exists
        if not os.path.exists(ptml_path):
            raise FileNotFoundError(f"The file {ptml_path} does not exist.")

        # Import process tree from PTML file
        pt = ptml_importer.apply(ptml_path)

        # Extract tree name from file path
        tree_name = os.path.splitext(os.path.basename(ptml_path))[0]

        # Save visualizations
        self._save_process_tree(pt, tree_name)
        self._save_petri_net(pt, tree_name)
    
    
if __name__ == "__main__":
    # Produce process trees and petri nets based on configuration
    # config_file = "./data_generation/analysis_params.json"
    # config = load_config(config_file)
    # generate_and_save_figures(config, save_path, {"bg_color": "white", "rankdir": "TB"})
    
    # Load and visualize a PTML file
    save_path = "./data_generation/visualizations"
    training_data_visualizer = TrainingDataVisualizer(save_path, {"bg_color": "white", "rankdir": "TB"})
    ptml_file_path = "./data_generation/synthetic_data/validation/pt_3500.ptml"
    training_data_visualizer.load_ptml_and_visualize(ptml_file_path)
