import torch
import time
from src.EventLog import EventLog
from src.GraphBuilder import GraphBuilder
from src.PetriNet import PetriNet, Place, Transition, Arc
from src.Models import GNNWithClassifier
from src.BatchFileLoader import BatchFileLoader
from torch_geometric.loader import DataLoader
from src.Trainer import Trainer
EPOCHS = 1
NUM_WORKERS = 2
BATCH_SIZE_LOAD = 10
BATCH_SIZE_TRAIN = 2
SHUFFLE = True
DATA_DIR = "./data_generation/synthetic_data/"
OUTPUT_MODEL_PATH = "models/graph_sage_model_with_dense_classifier.pth"



if __name__ == "__main__":
    trainer = Trainer(
        epochs=EPOCHS,
        batch_size_load=BATCH_SIZE_LOAD,
        batch_size_train=BATCH_SIZE_TRAIN,
        cpu_count=NUM_WORKERS,
        input_data=DATA_DIR,
        output_model_path=OUTPUT_MODEL_PATH
    )
    trainer.train()
