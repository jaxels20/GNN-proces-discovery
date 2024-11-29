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

class CandidateAnalyzer:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        

    def evaluate_candidate_places_on_single_pair(self, eventlog: EventLog, petrinet: PetriNet):
        """
        Evaluate the candidate places in the petrinet graph
        return the number of true positives, false positives, and false negatives
        """
        
        graphbuilder = GraphBuilder(eventlog)
        graph = graphbuilder.build_petrinet_graph()
        graph = self._select_all_places(graph)
        candidate_pn = PetriNet.from_graph(graph)
        tp, fp, fn  = compare_discovered_pn_to_true_pn(candidate_pn, petrinet)
        return tp, fp, fn
    
    def _select_all_places(self, graph: Data) -> Data:
        """set selected_nodes to true for all nodes in the graph
            and return the graph
        """
        graph["selected_nodes"] = torch.ones(graph.num_nodes, dtype=torch.bool)
        return graph
    
    def evaluate_candidate_places_on_all_pairs(self):
        """
        """
        
        # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        dataset_dirs = os.listdir(self.input_dir)
        # Filter out files, keep only directories
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
            petrinet = petrinets[id]
            tp, fp, fn = self.evaluate_candidate_places_on_single_pair(eventlog, petrinet)
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
            