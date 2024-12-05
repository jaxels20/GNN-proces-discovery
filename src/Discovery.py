""" This module contains the implementation of the discovery algorithms that are used to discover a Petri net from an event log. """

from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.Models import GNNWithClassifier
from gnn_miner.process_mining.process_discovery import GnnMiner   # GNN miner used in paper
import torch
from src.GraphBuilder import GraphBuilder
from src.inference import do_inference
from gnn_miner.data_handling.log import LogHandler

import os
import shutil
from pm4py.algo.discovery.alpha.algorithm import apply as pm4py_alpha_miner
from pm4py.algo.discovery.heuristics.algorithm import apply as pm4py_heuristic_miner
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
BEAM_WIDTH = 3
BEAM_LENGTH = 2
TOP_X_TRACES = 1000

class Discovery:
    @staticmethod
    def alpha_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the alpha miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_net, pm4py_initial_marking, pm4py_final_marking = pm4py_alpha_miner(pm4py_event_log)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net

    @staticmethod
    def heuristic_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the heuristic miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_net, pm4py_initial_marking, pm4py_final_marking = pm4py_heuristic_miner(pm4py_event_log)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net

    @staticmethod
    def inductive_miner(event_log: EventLog) -> PetriNet:
        """
        A wrapper for the inductive miner algorithm that is implemented in the pm4py library.
        """
        pm4py_event_log = event_log.to_pm4py()
        pm4py_process_tree = pm4py_inductive_miner(pm4py_event_log)
        pm4py_net, _, _ = pt_converter.apply(pm4py_process_tree)
        
        net = PetriNet.from_pm4py(pm4py_net)
        return net

    @staticmethod
    def AAU_miner(event_log: EventLog, model_path: str = "./models/graph_sage_model_with_dense_classifier.pth", eventually_follows_length: int = 1) -> PetriNet:
        # Load the model
        model = GNNWithClassifier(input_dim=64, hidden_dim=16, output_dim=1, dense_hidden_dim=32)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Get the event log as a graph
        graph_builder = GraphBuilder(eventlog=event_log, length=eventually_follows_length)
        graph = graph_builder.build_petrinet_graph()
        
        graph = do_inference(graph, model, device=torch.device('cpu'))
        discovered_pn = PetriNet.from_graph(graph)
        return discovered_pn
    
    @staticmethod
    def GNN_miner(event_log: EventLog, model_path: str = "./gnn_miner/ml_models/models/model_candidates_frequency_new_036.pth") -> PetriNet:
        # Define the temporary directory
        temp_dir = "./controlled_scenarios/temp_npz"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Create npz file of the event log
            pm4py_event_log = event_log.to_pm4py()
            log_handler = LogHandler('', fLog=pm4py_event_log)
            log_handler.getVariants()
            temp_npz_path = os.path.join(temp_dir, "npz_temp")
            log_handler.exportVariantsLog(temp_npz_path)

            # Load the model
            model = GnnMiner(fLogFilename=temp_npz_path, model_filename=model_path, embedding_size=21, embedding_strategy="onehot")

            # Discover the Petri net
            options = {
                "beam_width": BEAM_WIDTH,
                "beam_length": BEAM_LENGTH,
                "number_of_petrinets": 1,
                "export": "",
                "length_normalization": False,
                "timeout": None,
                "topXTraces": TOP_X_TRACES,
            }
            model.discover(conformance_check=False, **options)
            if model.mNet is not None:
                discovered_pn = PetriNet.from_pm4py(model.mNet)
                return discovered_pn
        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


    # Map method names to static methods
    methods = {
        "alpha": alpha_miner,
        "heuristic": heuristic_miner,
        "inductive": inductive_miner,
        "aau_miner": AAU_miner,
        "gnn_miner": GNN_miner
    }

    @classmethod
    def run_discovery(cls, method_name: str, event_log: EventLog, **kwargs) -> PetriNet:
        """
        Runs the specified discovery method based on method_name.
        
        Parameters:
        - method_name (str): The name of the method to run (e.g., "alpha", "heuristic", "inductive", "aau_miner", "gnn_miner").
        - event_log (EventLog): The event log to be passed to the discovery method.
        - **kwargs: Additional arguments to be passed to the method, such as model_path for GNN_miner.
        
        Returns:
        - PetriNet: The discovered Petri net.
        """
        method = cls.methods.get(method_name)
        if method is None:
            raise ValueError(f"Discovery method '{method_name}' not found.")
        
        # Call the appropriate method, passing **kwargs for any extra arguments needed
        return method(event_log, **kwargs)
    
    
    
    