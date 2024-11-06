from util import load_all_eventlogs
from Evaluator import MultiEvaluator

if __name__ == "__main__":
    dataset_dir = "./real_life_datasets"
    all_eventlogs = load_all_eventlogs(dataset_dir)
    
    # Create and evaluate the MultiEvaluator
    multi_evaluator = MultiEvaluator(all_eventlogs)
    results_df = multi_evaluator.evaluate_all(output_png=True)
    results_df.to_csv("./results/results.csv")
    multi_evaluator.save_dataframe_to_pdf(results_df, "./results/results.pdf")
    
    
    
    

