""" This module contains the implementation of the discovery algorithms that are used to discover a Petri net from an event log. """

from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.Models import GNNWithClassifier
import torch
from src.GraphBuilder import GraphBuilder
from src.inference import do_inference

from pm4py.algo.discovery.alpha.algorithm import apply as pm4py_alpha_miner
from pm4py.algo.discovery.heuristics.algorithm import apply as pm4py_heuristic_miner
from pm4py.algo.discovery.inductive.algorithm import apply as pm4py_inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter


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
    def GNN_miner(event_log: EventLog, model_path: str = "./models/graph_sage_model_with_dense_classifier.pth", eventually_follows_length: int = 1) -> PetriNet:
        # Load the model
        model = GNNWithClassifier(input_dim=64, hidden_dim=16, output_dim=1, dense_hidden_dim=32)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Get the event log as a graph
        graph_builder = GraphBuilder(eventlog=event_log, length=eventually_follows_length)
        graph = graph_builder.build_petrinet_graph()
        
        graph = do_inference(graph, model)
        discovered_pn = PetriNet.from_graph(graph)
        return discovered_pn

    # Map method names to static methods
    methods = {
        "alpha": alpha_miner,
        "heuristic": heuristic_miner,
        "inductive": inductive_miner,
        "gnn": GNN_miner
    }

    @classmethod
    def run_discovery(cls, method_name: str, event_log: EventLog, **kwargs) -> PetriNet:
        """
        Runs the specified discovery method based on method_name.
        
        Parameters:
        - method_name (str): The name of the method to run (e.g., "alpha", "heuristic", "inductive", "GNN").
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
    
    
    
    
    
    
    