from src.BatchFileLoader import BatchFileLoader
from gnn_miner.process_mining.process_discovery import GnnMiner
from src.Evaluator import MultiEvaluator
import os
INPUT_DIR = "./controlled_scenarios/" # Assume structered like this "./controlled_scenarios/dataset_name/"
OUTPUT_DIR = "./controlled_scenarios_results/" 
METHODS = ["gnn_miner"]
METHODS = ["alpha", "heuristic", "inductive", "gnn_miner", "aau_miner"]
NUM_WORKERS = 1

if __name__ == "__main__":
    dataset_dirs = os.listdir(INPUT_DIR)
    # remove all files from the list
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{INPUT_DIR}{x}")]
    eventlogs = {} # name: eventlog
    loader = BatchFileLoader(cpu_count=1)
    for dataset_dir in dataset_dirs:
        temp_eventlogs = loader.load_all_eventlogs(f"{INPUT_DIR}{dataset_dir}")
        # check that there is only one event log in the directory
        assert len(temp_eventlogs) == 1, f"Expected one event log in {dataset_dir}, got {len(temp_eventlogs)}"
        key = dataset_dir
        eventlogs[key] = next(iter(temp_eventlogs.values()))
    
    #Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(eventlogs, methods=METHODS)
    results_df = multi_evaluator.evaluate_all(num_cores=NUM_WORKERS)
    results_df.to_csv(OUTPUT_DIR + "results.csv")
    multi_evaluator.save_dataframe_to_pdf(results_df, OUTPUT_DIR + "results.pdf")
    multi_evaluator.export_petri_nets(OUTPUT_DIR, format="png")