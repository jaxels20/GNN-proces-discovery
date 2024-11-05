from PetriNet import PetriNet
from EventLog import EventLog
from pm4py.algo.evaluation.replay_fitness.algorithm import apply as replay_fitness
from pm4py.algo.evaluation.precision.algorithm import apply as precision
from pm4py.algo.evaluation.generalization.algorithm import apply as generalization
from pm4py.algo.evaluation.simplicity.algorithm import apply as simplicity
import pandas as pd
from inference import discover
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
            "replay_fitness": self.get_replay_fitness(),
            "precision": self.get_precision(),
        }
        data["f1_score"] = self.get_f1_score(data["precision"], data["replay_fitness"])
        
        return data    
    
    def get_simplicity(self):
        simplicity_value = simplicity(self.process_model_pm4py)
        return simplicity_value
    
    def get_generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return generalization_value
    
    def get_replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return fitness['percFitTraces']
    
    def get_precision(self):
        precision_value = precision(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return precision_value
    
    def get_f1_score(self, precision=None, fitness=None):
        if precision is None:
            precision = self.get_precision()
        if fitness is None:
            fitness = self.get_replay_fitness()
        
        f1_score = 2 * (precision * fitness) / (precision + fitness)
        return f1_score
        

class MultiEvaluator:
    def __init__(self, event_logs: dict):
        """
        Initialize with dictionaries of Petri nets and event logs.
        """
        self.petri_nets = {i: discover(event_logs[i]) for i in event_logs}
        self.event_logs = event_logs

    def evaluate_all(self):
        """
        Evaluate all Petri nets against their corresponding event logs and return a DataFrame.
        """
        results = []

        for key in self.petri_nets:
            if key in self.event_logs:
                # Initialize the evaluator for the current Petri net and event log
                evaluator = SingleEvaluator(self.petri_nets[key], self.event_logs[key])

                # Get metrics and add the ID for identification
                metrics = evaluator.get_evaluation_metrics()
                # round the values to 4 decimal places
                metrics = {k: round(v, 4) for k, v in metrics.items()}
                
                metrics['id'] = key
                
                # Append the results
                results.append(metrics)
            else:
                print(f"No matching event log for Petri net with ID {key}")

        # Convert the results to a DataFrame
        return pd.DataFrame(results)

    def save_dataframe_to_pdf(self, df, pdf_path):
        # Set up PDF document
        with PdfPages(pdf_path) as pdf:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))  # Adjust figure size as needed
            ax.axis('tight')
            ax.axis('off')

            # Create the table
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)  # Scale table to fit PDF page nicely

            # Save the figure containing the table to the PDF
            pdf.savefig(fig)
            plt.close()
    
    


    