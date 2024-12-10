from src.PetriNet import PetriNet
from src.EventLog import EventLog
#from pm4py.algo.evaluation.replay_fitness.algorithm import apply as replay_fitness
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import apply as replay_fitness
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as precision
from pm4py.algo.evaluation.generalization.variants.token_based import apply as generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as simplicity
import pandas as pd
from src.Discovery import Discovery
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor
import copy
import tabulate


# This class can evaluate a discovered process model against an event log (only one!)
class SingleEvaluator:
    def __init__(self, proces_model: PetriNet, eventlog: EventLog):
        self.process_model = proces_model
        self.eventlog = eventlog
        
        # convert the process model to pm4py format
        self.process_model_pm4py, self.init_marking, self. final_marking = self.process_model.to_pm4py()

        # convert the eventlog to pm4py format
        self.event_log_pm4py = self.eventlog.to_pm4py()
    
    def get_evaluation_metrics(self):
        data = {
            "simplicity": self.get_simplicity(),
            "generalization": self.get_generalization(),
            **self.get_replay_fitness(),
            "precision": self.get_precision(),
        }
        data["f1_score"] = self.get_f1_score(data["precision"], data["log_fitness"])
        return data    
    
    def get_simplicity(self):
        simplicity_value = simplicity(self.process_model_pm4py)
        return simplicity_value
    
    def get_generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return generalization_value
    
    def get_replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return fitness
    
    def get_precision(self):
        precision_value = precision(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return precision_value
    
    def get_f1_score(self, precision=None, fitness=None):
        if precision is None:
            precision = self.get_precision()
        if fitness is None:
            fitness = self.get_replay_fitness()
        try:
            f1_score = 2 * (precision * fitness) / (precision + fitness)
        except ZeroDivisionError:
            f1_score = 0.0
        return f1_score
        

# Define a helper function that will handle evaluation for a single Petri net and event log pair
def evaluate_single(miner: str, dataset: str, petri_net: PetriNet, event_log: EventLog):
    evaluator = SingleEvaluator(petri_net, event_log)
    
    # Get metrics and round to 4 decimal places
    metrics = {k: round(v, 3) for k, v in evaluator.get_evaluation_metrics().items()}
    metrics['miner'] = miner
    metrics['dataset'] = dataset
    
    
    return metrics

# This function discovers a process model from an event log 
# and evaluates it against the event log (calculates the metrics)
class MultiEvaluator:
    def __init__(self, event_logs: dict, methods: list):
        """
        Initialize with dictionaries of Petri nets and event logs.
        Args:
        - event_logs (dict): A dictionary where keys are event log names and values are EventLog objects.
        
        - methods (list): A list of strings representing the discovery methods to use. can be "alpha", "heuristic", "inductive", "GNN"
        
        """
        self.event_logs = event_logs # dictionary of event logs with keys as event log names and values as EventLog objects
        self.petri_nets = {method: {} for method in methods} # dictionary of Petri nets with keys as discovery methods 
        # and values a dict of event log names and PetriNet objects
        for method in methods:
            for event_log_name, event_log in self.event_logs.items():
                pn_result = Discovery.run_discovery(method, event_log)
                if pn_result is not None:   # check needed since gnn miner sometimes doesnt work => returns None
                    self.petri_nets[method][event_log_name] = pn_result
        
    def evaluate_all(self, num_cores=None):
            """
            Evaluate all Petri nets against their corresponding event logs using multiprocessing,
            and return a DataFrame with metrics.
            """
            results = []
            
            # Use ProcessPoolExecutor for multiprocessing
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                
                # Iterate through each miner type and dataset in petri_nets
                for miner, datasets in self.petri_nets.items():
                    for dataset, petri_net in datasets.items():
                        if dataset in self.event_logs:
                            event_log = self.event_logs[dataset]
                            futures.append(
                                executor.submit(evaluate_single, miner, dataset, petri_net, event_log)
                            )
                
                # Collect the results as they complete
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Error evaluating Petri net: {e}")
            
            # Convert the list of dictionaries to a DataFrame
            return pd.DataFrame(results)

    def export_petri_nets(self, output_dir, format="png"):
        """
        Export all Petri nets to the specified directory. They format can be specified as "png" or "pdf.
        """
        for miner, datasets in self.petri_nets.items():
            for dataset, petri_net in datasets.items():
                if format == "png":
                    petri_net.visualize(f"{output_dir}/{dataset}/{miner}", format="png")
                elif format == "pdf":
                    petri_net.visualize(f"{output_dir}/{dataset}/{miner}", format="pdf")
                else:
                    print(f"Invalid format: {format}. Must be 'png' or 'pdf'.")
                    break

    def save_df_to_pdf(self, df, pdf_path):
        """
        Save the DataFrame to a single PDF figure with all datasets grouped.
        """
        table_data = []
        column_headers = ["Dataset", "Method", "F1-Score", "Fitness", "Precision", "Generalization", "Simplicity"]
        grouped = df.groupby('dataset')
        
        for dataset, group in grouped:
            # Add the dataset name in the first row
            for i, (_, row) in enumerate(group.iterrows()):
                if i == 0:
                    table_data.append([
                        dataset,  # Dataset name printed only in the first row
                        row['miner'],
                        f"{row['f1_score']:.3f}",
                        f"{row['log_fitness']:.3f}",
                        f"{row['precision']:.3f}",
                        f"{row['generalization']:.3f}",
                        f"{row['simplicity']:.3f}",
                    ])
                else:
                    table_data.append([
                        "",  # Empty dataset name for subsequent rows
                        row['miner'],
                        f"{row['f1_score']:.3f}",
                        f"{row['log_fitness']:.3f}",
                        f"{row['precision']:.3f}",
                        f"{row['generalization']:.3f}",
                        f"{row['simplicity']:.3f}",
                    ])

        # Create the figure and add the table
        fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.3))
        ax.axis('off')

        # Add the table to the figure
        table = ax.table(
            cellText=table_data,
            colLabels=column_headers,
            loc='center',
            cellLoc='center',
            colLoc='center',
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(column_headers))))

        # Save the figure to the PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF saved as {pdf_path}")
        
    def save_df_to_latex(self, df: pd.DataFrame, output_dir: str, scenario: str) -> None:
        """
        Save the DataFrame to a LaTeX table format.
        """
        # Rename the datasets to avoid problems with underscores in LaTeX
        if scenario == "controlled":
            df['dataset'] = df['dataset'].replace({
                'overleaf_example': 'Overleaf Example',
                'simple_sequence': 'Simple Sequence',
                'long_dependency': 'Long Dependency',
                'loop_lenght_1': 'Loop Length 1',
                'loop_lenght_2': 'Loop Length 2',
                'simple_and_split': 'Simple AND Split',
                'simple_xor_split': 'Simple XOR Split'
            })
        
        # Group by dataset and miner
        grouped = df.groupby(['dataset', 'miner'])
        table_data = []
        previous_dataset = None

        for (dataset, miner), group in grouped:
            # Add a horizontal line between groups
            if dataset != previous_dataset and previous_dataset is not None:
                table_data.append(r"\hline")
            
            # Create a row for the current group
            row = (
                f"{dataset if dataset != previous_dataset else ''} & "
                f"{miner} & "
                f"{group['f1_score'].mean():.3f} & "
                f"{group['log_fitness'].mean():.3f} & "
                f"{group['precision'].mean():.3f} & "
                f"{group['generalization'].mean():.3f} & "
                f"{group['simplicity'].mean():.3f} \\\\"
            )
            table_data.append(row)
            previous_dataset = dataset

        # Add a final horizontal line
        table_data.append(r"\hline")

        # Construct the LaTeX table
        latex_table = (
            r"\begin{tabular}{|l|l|r|r|r|r|r|}" + "\n"
            r"\hline" + "\n"
            r" Dataset & Method & F1-score & Fitness & Precision & Generalization & Simplicity \\" + "\n"
            r"\hline" + "\n"
            + "\n".join(table_data) + "\n"
            r"\end{tabular}"
        )

        # Save the table to a file
        with open(output_dir + "latex_table.tex", "w") as f:
            f.write(latex_table)

        # print(latex_table)
