import torch
import time
from torch.utils.data import DataLoader
from Models import GNNWithClassifier
from GraphBuilder import GraphBuilder
from BatchFileLoader import BatchFileLoader


class Trainer:
    def __init__(
        self,
        eventlog_dir,
        petrinet_dir,
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
        self.eventlog_dir = eventlog_dir
        self.petrinet_dir = petrinet_dir
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
            "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
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
            pos_weight=torch.tensor([50.0], device=self.device)
        )

    def train_epoch(self, batch_loader):
        epoch_loss = 0
        self.model.train()

        for eventlog_batch, petrinet_batch in zip(
            batch_loader.batch_eventlog_loader(self.eventlog_dir, self.batch_size_load),
            batch_loader.batch_petrinet_loader(self.petrinet_dir, self.batch_size_load),
        ):
            pyg_graphs = []

            # Build and annotate graphs for the current batch
            for id, petrinet in petrinet_batch.items():
                if id in eventlog_batch:  # Ensure matching EventLog exists
                    eventlog = eventlog_batch[id]
                    graph_builder = GraphBuilder(eventlog)
                    graph = graph_builder.build_petrinet_graph()
                    graph = graph_builder.annotate_petrinet_graph(graph, petrinet)
                    pyg_graphs.append(graph)

            # Create a DataLoader for the current batch of graphs
            data_loader = DataLoader(
                pyg_graphs,
                batch_size=self.batch_size_train,
                shuffle=True,  # You can make `shuffle` configurable
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
        start_time = time.time()

        batch_loader = BatchFileLoader(self.cpu_count)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_loss = self.train_epoch(batch_loader)

            # Print average loss per epoch
            avg_loss = epoch_loss / self.epochs  # Adjust as needed based on batch size
            epoch_end = time.time()
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Time: {epoch_end - epoch_start:.2f}s")

        # Calculate the total training time
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        # Save the trained model
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.output_model_path)
        print(f"Model saved to '{self.output_model_path}'")
