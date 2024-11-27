import os
import csv
from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.GraphBuilder import GraphBuilder
from src.Comparison import compare_discovered_pn_to_true_pn
from torch_geometric.data import Data
import torch

INPUT_DIR = "./controlled_scenarios/"  # Assume structured like this "./controlled_scenarios/dataset_name/"
OUTPUT_DIR = "./con_scenarios_candidate_analysis_results/"  # Directory to save CSV file
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")


def select_all_places(graph: Data) -> Data:
    """Set selected_nodes to true for all nodes in the graph."""
    graph["selected_nodes"] = torch.ones(graph.num_nodes, dtype=torch.bool)
    return graph


class FileLoader:
    @staticmethod
    def _load_eventlog(file_path: str):
        """Load a single EventLog object from a file."""
        if not file_path.endswith(".xes"):
            return None  # Skip non-XES files
        try:
            el = EventLog.load_xes(file_path)
            base_name = os.path.basename(file_path)
            if "_" in base_name:
                file_id = base_name.split("_")[1].removesuffix(".xes")
            else:
                file_id = base_name.removesuffix(".xes")
        except Exception as e:
            raise ValueError(f"Failed to load event log from {file_path}: {e}")
        return file_id, el

    def load_eventlog(self, directory):
        """Load the eventlog.xes file from the specified directory."""
        eventlog_path = os.path.join(directory, "eventlog.xes")
        if not os.path.isfile(eventlog_path):
            raise FileNotFoundError(f"eventlog.xes not found in {directory}")
        return self._load_eventlog(eventlog_path)

    def load_petrinet(self, directory):
        """Load the petri_net.ptml file from the specified directory."""
        petrinet_path = os.path.join(directory, "petri_net.ptml")
        if not os.path.isfile(petrinet_path):
            raise FileNotFoundError(f"petri_net.ptml not found in {directory}")
        try:
            pn = PetriNet.from_ptml(petrinet_path)
        except Exception as e:
            raise ValueError(f"Failed to load Petri net from {petrinet_path}: {e}")
        return pn


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

    dataset_dirs = os.listdir(INPUT_DIR)
    # Filter out files, keep only directories
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(os.path.join(INPUT_DIR, x))]
    
    eventlogs = {}  # name: eventlog
    petrinets = {}  # name: petrinet
    loader = FileLoader()

    # Load all eventlogs and petrinets
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(INPUT_DIR, dataset_dir)
        try:
            # Load eventlog
            file_id, eventlog = loader.load_eventlog(dataset_path)
            # Load petrinet
            petrinet = loader.load_petrinet(dataset_path)
            # Add to dictionaries
            eventlogs[dataset_dir] = eventlog
            petrinets[dataset_dir] = petrinet
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {dataset_dir}: {e}")

    # Ensure we have matching eventlog and petrinet pairs
    matching_scenarios = set(eventlogs.keys()) & set(petrinets.keys())

    # Prepare results
    results = []
    for scenario in matching_scenarios:
        try:
            eventlog = eventlogs[scenario]
            petrinet = petrinets[scenario]

            # Perform comparison
            graphbuilder = GraphBuilder(eventlog)
            graph = graphbuilder.build_petrinet_graph()
            graph = select_all_places(graph)
            candidate_pn = PetriNet.from_graph(graph)
            true_positives, false_positives, false_negatives = compare_discovered_pn_to_true_pn(candidate_pn, petrinet)

            # Append result for this scenario
            results.append({"Scenario": scenario, "True_positives": true_positives, "False_positives": false_positives, "False_negatives": false_negatives, "Precision": true_positives / (true_positives + false_positives), "Recall": true_positives / (true_positives + false_negatives)})
        except Exception as e:
            print(f"Error processing scenario {scenario}: {e}")

    # Save results to CSV
    with open(RESULTS_FILE, mode="w", newline="") as csvfile:
        fieldnames = ["Scenario", "True_positives", "False_positives", "False_negatives", "Precision", "Recall"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()  # Write column names
        writer.writerows(results)  # Write scenario results

    print(f"Results saved to {RESULTS_FILE}")
