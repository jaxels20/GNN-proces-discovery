from src.EventLog import EventLog
from src.PetriNet import PetriNet
from src.GraphBuilder import GraphBuilder
from src.Comparison import compare_discovered_pn_to_true_pn
from torch_geometric.data import Data
import torch
import os
from src.BatchFileLoader import BatchFileLoader
from src.Comparison import compare_discovered_pn_to_true_pn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pm4py.objects.petri_net.utils.reduction import apply_fsp_rule

class CandidateAnalyzer:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
    def evaluate_candidate_places(self, eventlog: EventLog, true_petrinet: PetriNet, add_silent_transitions: bool = False, export_nets=False, id=None):
        """
        Evaluate the candidate places in the petrinet graph.
        If export_nets is True, export the petrinet graphs to the output directory with the given id.
        Returns the number of true positives, false positives, and false negatives
        """
        
        graphbuilder = GraphBuilder(eventlog)
        graph = graphbuilder.build_petrinet_graph()
        if graph is None:
            return None, None, None
        graph = self._select_all_places(graph)
        candidate_pn = PetriNet.from_graph(graph)
        if add_silent_transitions:
            candidate_pn.add_silent_transitions(eventlog)
        
        if export_nets and id is not None:
            true_petrinet.visualize(os.path.join(self.output_dir, f"{id}_true_petrinet")) 
            candidate_pn.visualize(os.path.join(self.output_dir, f"{id}_candidate_petrinet"))
        
        tp, fp, fn = compare_discovered_pn_to_true_pn(candidate_pn, true_petrinet)
        return tp, fp, fn
    
    def _select_all_places(self, graph: Data) -> Data:
        """set selected_nodes to true for all nodes in the graph
            and return the graph
        """
        graph["selected_nodes"] = torch.ones(graph.num_nodes, dtype=torch.bool)
        return graph
    
    def evaluate_on_controlled_scenarios(self, add_silent_transitions: bool = False):
        # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
        # Filter out files, keep only directories
        dataset_dirs = os.listdir(self.input_dir)     
        dataset_dirs = [x for x in dataset_dirs if not os.path.isfile(os.path.join(self.input_dir, x))]
        
        eventlogs = {}  # name: eventlog
        petrinets = {}  # name: petrinet

        # Load all eventlogs and petrinets
        for dataset_dir in dataset_dirs:
            dataset_path = os.path.join(self.input_dir, dataset_dir)
            try:
                loader = BatchFileLoader(1)
                eventlog = loader.load_all_eventlogs(dataset_path)
                petrinet = loader.load_all_petrinets(dataset_path)

                # assert that they are of length 1
                assert len(eventlog) == 1
                assert len(petrinet) == 1
                
                eventlogs[dataset_dir] = next(iter(eventlog.values()))
                petrinets[dataset_dir] = next(iter(petrinet.values()))

                
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping {dataset_dir}: {e}")
                
        
        results = []
                
        for id, eventlog in eventlogs.items():
            print(f"Evaluating {id}")
            petrinet = petrinets[id]
            tp, fp, fn = self.evaluate_candidate_places(eventlog, petrinet, add_silent_transitions, export_nets=True, id=id)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            results.append(
                {"Scenario": id.replace("_", " "),
                 "True_positives": tp, 
                 "False_positives": fp, 
                 "False_negatives": fn, 
                 "Precision": round(precision, 3), 
                 "Recall": round(recall, 3)}
                )
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        print(f"Results saved to {os.path.join(self.output_dir, 'results.csv')}")
        
        self.save_df_to_pdf(df, os.path.join(self.output_dir, "results.pdf"))
        print(f"Results saved to {os.path.join(self.output_dir, 'results.pdf')}")
        
        self.save_df_to_latex(df, os.path.join(self.output_dir, "results.tex"))
        print(f"Results saved to {os.path.join(self.output_dir, 'results.tex')}")
     
    def evaluate_on_synthetic_data(self):
         # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        loader = BatchFileLoader(1)
        eventlogs = loader.load_all_eventlogs(self.input_dir)
        petrinets = loader.load_all_petrinets(self.input_dir)
        
        results = []
        
        for id, eventlog in eventlogs.items():
            petrinet = petrinets[id]
            tp, fp, fn = self.evaluate_candidate_places(eventlog, petrinet, export_nets=True, id=id)
            if any(x is None for x in [tp, fp, fn]):
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            results.append(
                {"Scenario": id, 
                 "True_positives": tp, 
                 "False_positives": fp, 
                 "False_negatives": fn, 
                 "Precision": precision, 
                 "Recall": recall}
                )
        
        df = pd.DataFrame(results)
        
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        print(f"Results saved to {os.path.join(self.output_dir, 'results.csv')}")
        
        self.save_df_to_pdf(df, os.path.join(self.output_dir, "results.pdf"))
        print(f"Results saved to {os.path.join(self.output_dir, 'results.pdf')}")
        
        self.save_df_to_latex(df, os.path.join(self.output_dir, "results.tex"))
        print(f"Results saved to {os.path.join(self.output_dir, 'results.tex')}")
        
        # Mean precision and recall for all scenarios
        mean_precision = df["Precision"].mean()
        mean_recall = df["Recall"].mean()
        print(f"Mean Precision: {mean_precision}")
        print(f"Mean Recall: {mean_recall}")
        
    def save_df_to_pdf(self, df: pd.DataFrame, pdf_path: str):
        # sort df by scenario name
        df = df.sort_values(by="Scenario")

        # Step 1: Create a PDF file with pages for each group or the entire dataset
        with PdfPages(pdf_path) as pdf:
            # Create a figure for the entire dataset
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')  # Turn off the axis
            
            # Add a title for the dataset
            fig.suptitle("Results Table", fontsize=14, fontweight='bold')
            
            # Format the DataFrame data as a table
            table_data = df.values
            column_headers = df.columns
            
            # Add the table to the figure
            ax.table(cellText=table_data, colLabels=column_headers, loc='center', cellLoc='center')
            
            # Adjust layout and save this page to the PDF
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    def save_df_to_latex(self, df: pd.DataFrame, latex_path: str):
        df = df.sort_values(by="Scenario")
        df.to_latex(latex_path, bold_rows=True, index=False)