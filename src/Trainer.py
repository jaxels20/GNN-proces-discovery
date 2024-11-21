import torch
import time
from torch_geometric.loader import DataLoader
from src.GraphBuilder import GraphBuilder
from src.BatchFileLoader import BatchFileLoader
from src.Models import GNNWithClassifier
from src.inference import do_inference
from src.Comparison import compare_discovered_pn_to_true_pn
from src.PetriNet import PetriNet
import pandas as pd
from src.TrainingFigureGenerator import TrainingFigureGenerator
from copy import deepcopy


class Trainer:
    def __init__(
        self,
        input_data,
        input_dim=64,
        hidden_dim=16,
        dense_hidden_dim=32,
        output_dim=1,
        lr=0.01,
        epochs=100,
        batch_size_train=10,
        batch_size_load=100,
        cpu_count=1,
        output_model_path="model.pth",
    ):
        self.train_data_dir = input_data + "train/"
        self.test_data_dir = input_data + "test/"
        self.validation_data_dir = input_data + "validation/"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dense_hidden_dim = dense_hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_load = batch_size_load
        self.cpu_count = cpu_count
        self.output_model_path = output_model_path

        # Set device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Training on device: {self.device}")

        # Initialize model, optimizer, and loss function
        self.model = GNNWithClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dense_hidden_dim=self.dense_hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([10.0], device=self.device)
        )

    def train_epoch(self, batch_loader):
        epoch_loss = 0
        self.model.train()

        for eventlog_batch, petrinet_batch in zip(
            batch_loader.batch_eventlog_loader(
                self.train_data_dir, self.batch_size_load
            ),
            batch_loader.batch_petrinet_loader(
                self.train_data_dir, self.batch_size_load
            ),
        ):
            pyg_graphs = []

            # Build and annotate graphs for the current batch
            for id, petrinet in petrinet_batch.items():
                if id in eventlog_batch:  # Ensure matching EventLog exists
                    eventlog = eventlog_batch[id]
                    graph_builder = GraphBuilder(eventlog)
                    graph = graph_builder.build_petrinet_graph()
                    if graph is None:
                        continue
                    graph = graph_builder.annotate_petrinet_graph(graph, petrinet)
                    pyg_graphs.append(graph)

            # Create a DataLoader for the current batch of graphs
            data_loader = DataLoader(
                pyg_graphs,
                batch_size=self.batch_size_train,
                shuffle=True,
                pin_memory=True,
                num_workers=self.cpu_count,
            )

            # Train on the current batch
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                out = self.model(batch.node_x, batch.edge_index)

                # Compute loss only for place nodes
                loss = self.criterion(
                    out[batch["place_mask"]], batch.labels[batch["place_mask"]]
                )
                print(f"Batch {batch_idx}, Avg Loss: {loss:.4f}")

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

        return epoch_loss

    def train(self):
        print(f"Starting training for {self.epochs} epochs")
        training_stats = {
            "epoch": [],
            "avg_loss": [],
            "time": [],
            "avg_true_positives": [],
            "avg_false_positives": [],
            "avg_false_negatives": [],
        }
        start_time = time.time()

        batch_loader = BatchFileLoader(self.cpu_count)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_loss = self.train_epoch(batch_loader)

            # Print average loss per epoch
            avg_loss = epoch_loss / self.epochs  # Adjust as needed based on batch size
            epoch_end = time.time()

            # Test on validation data
            tp, fp, fn = self.test_on_validation()
            training_stats["epoch"].append(epoch)
            training_stats["avg_loss"].append(avg_loss)
            training_stats["time"].append(epoch_end - epoch_start)
            training_stats["avg_true_positives"].append(tp)
            training_stats["avg_false_positives"].append(fp)
            training_stats["avg_false_negatives"].append(fn)
            print(
                f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Time: {epoch_end - epoch_start:.2f} seconds, TP: {tp}, FP: {fp}, FN: {fn}"
            )

        # Calculate the total training time
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        # Save the trained model
        self.save_model()

        # # save the training stats as a csv file
        # training_stats_df = pd.DataFrame(training_stats)
        # training_stats_df.to_csv("training_stats.csv", index=False)

        # create a training figure
        fig, ax = TrainingFigureGenerator.create_training_figure(training_stats)
        fig.savefig("./training_figure.png")

    def save_model(self):
        torch.save(self.model.state_dict(), self.output_model_path)
        print(f"Model saved to '{self.output_model_path}'")

    def test_on_validation(self):
        loader = BatchFileLoader(self.cpu_count)
        eventlogs = loader.load_all_eventlogs(self.validation_data_dir)
        petrinets = loader.load_all_petrinets(self.validation_data_dir)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        num_graphs = 0

        self.model.eval()
        with torch.no_grad():
            for id, petrinet in petrinets.items():
                if id in eventlogs:
                    eventlog = eventlogs[id]
                    graph_builder = GraphBuilder(eventlog)
                    graph = graph_builder.build_petrinet_graph()
                    if graph is None:
                        continue

                    ground_truth_graph = graph_builder.annotate_petrinet_graph(
                        graph, petrinet
                    )
                    ground_truth_pn = PetriNet.from_graph(ground_truth_graph)
                    # discovered graph
                    discovered_graph = do_inference(
                        graph, deepcopy(self.model), self.device
                    )
                    discovered_pn = PetriNet.from_graph(discovered_graph)
                    # Compare the discovered graph with the ground truth graph
                    true_positives, false_positives, false_negatives = (
                        compare_discovered_pn_to_true_pn(discovered_pn, ground_truth_pn)
                    )

                    # Calculate precision, recall, and F1 score
                    # precision = true_positives / (true_positives + false_positives)
                    # recall = true_positives / (true_positives + false_negatives)
                    # try: # Avoid division by zero
                    #     f1_score = 2 * (precision * recall) / (precision + recall)
                    # except ZeroDivisionError:
                    #     f1_score = 0.0
                    total_tp += true_positives
                    total_fp += false_positives
                    total_fn += false_negatives
                    num_graphs += 1
        self.model.train()

        # TP: The places that are present in both the discovered and ground truth Petri nets
        # FP: The places that are present in the discovered Petri net but not in the ground truth Petri net
        # FN: The places that are present in the ground truth Petri net but not in the discovered Petri net

        return total_tp / num_graphs, total_fp / num_graphs, total_fn / num_graphs
