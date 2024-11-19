from src.BatchFileLoader import BatchFileLoader
from src.Evaluator import MultiEvaluator
import os

if __name__ == "__main__":
    input_dir = "./controlled_scenarios/" # Assume structered like this "./controlled_scenarios/dataset_name/" 
    output_dir = "./controlled_scenarios_results/" 
    dataset_dirs = os.listdir(input_dir)
    # remove all files from the list
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{input_dir}{x}")]
    eventlogs = {} # name: eventlog
    loader = BatchFileLoader(cpu_count=4)
    for dataset_dir in dataset_dirs:
        temp_eventlogs = loader.load_all_eventlogs(f"{input_dir}{dataset_dir}")
        # check that there is only one event log in the directory
        assert len(temp_eventlogs) == 1, f"Expected one event log in {dataset_dir}, got {len(temp_eventlogs)}"
        key = dataset_dir
        eventlogs[key] = next(iter(temp_eventlogs.values()))
    
    #Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(eventlogs, methods=["alpha", "heuristic", "inductive", "gnn"])
    results_df = multi_evaluator.evaluate_all(num_cores=4)
    results_df.to_csv("./controlled_scenarios_results/results.csv")
    multi_evaluator.save_dataframe_to_pdf(results_df, "./controlled_scenarios_results/results.pdf")
    multi_evaluator.export_petri_nets(output_dir, format="png")