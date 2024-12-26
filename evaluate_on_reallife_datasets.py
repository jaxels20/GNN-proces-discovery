from src.BatchFileLoader import BatchFileLoader
from src.Evaluator import MultiEvaluator
METHODS = ["Alpha", "Heuristic", "Inductive", "GNN Miner", "AAU Miner"]
NUM_WORKERS = 4

if __name__ == "__main__":
    dataset_dir = "./real_life_datasets"
    loader = BatchFileLoader(cpu_count=NUM_WORKERS)
    all_eventlogs = loader.load_all_eventlogs(dataset_dir)
    
    # Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(all_eventlogs, METHODS)
    results_df = multi_evaluator.evaluate_all(num_cores=NUM_WORKERS)
    results_df.to_csv("./real_life_results/results.csv")
    multi_evaluator.save_df_to_pdf(results_df, "./real_life_results/results.pdf")    
    multi_evaluator.export_petri_nets("./real_life_results")