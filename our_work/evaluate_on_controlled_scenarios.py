from util import load_all_eventlogs
from Evaluator import MultiEvaluator
import os

if __name__ == "__main__":
    input_dir = "./controlled_scenarios/" # Assume structered like this "./controlled_scenarios/dataset_name/" 
    dataset_dirs = os.listdir(input_dir)
    # remove all files from the list
    dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(f"{input_dir}{x}")]
    eventlogs = {} # name: eventlog
    for dataset_dir in dataset_dirs:
        temp_eventlogs = load_all_eventlogs(f"{input_dir}{dataset_dir}")
        # check that there is only one event log in the directory
        assert len(temp_eventlogs) == 1, f"Expected one event log in {dataset_dir}, got {len(temp_eventlogs)}"
        
        # add the construct the key as input_dir/dataset_dir/
        key = f"{input_dir}{dataset_dir}/discovered_petri_net"

        eventlogs[key] = next(iter(temp_eventlogs.values()))
        
    #Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(eventlogs)
    results_df = multi_evaluator.evaluate_all(output_png=True, num_cores=1)
    #results_df.to_csv("./results/results.csv")
    #multi_evaluator.save_dataframe_to_pdf(results_df, "./results/results.pdf")