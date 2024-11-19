from src.BatchFileLoader import BatchFileLoader
from src.Evaluator import MultiEvaluator

if __name__ == "__main__":
    dataset_dir = "./real_life_datasets"
    loader = BatchFileLoader(cpu_count=4)
    all_eventlogs = loader.load_all_eventlogs(dataset_dir)
    
    # appned "./results" to the keys of the dictionary
    #all_eventlogs = {f"./real_life_results/{k}": v for k, v in all_eventlogs.items()}
    
    # Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(all_eventlogs, ["alpha", "heuristic", "inductive"])
    
    
    results_df = multi_evaluator.evaluate_all(num_cores=4)
    
    results_df.to_csv("./real_life_results/results.csv")
    
    multi_evaluator.save_dataframe_to_pdf(results_df, "./real_life_results/results.pdf")
    
    multi_evaluator.export_petri_nets("./real_life_results")
    
    

