import torch
import time
from src.EventLog import EventLog
from src.GraphBuilder import GraphBuilder
from src.PetriNet import PetriNet, Place, Transition, Arc
from src.Models import GNNWithClassifier
from src.BatchFileLoader import BatchFileLoader
from torch_geometric.loader import DataLoader
from src.Trainer import Trainer
EPOCHS = 5
NUM_WORKERS = 14
BATCH_SIZE_LOAD = 500
BATCH_SIZE_TRAIN = 50
SHUFFLE = True
EVENTLOG_DIR = "./data_generation/synthetic_data"
PETRINET_DIR = "./data_generation/synthetic_data"
OUTPUT_MODEL_PATH = "models/graph_sage_model_with_dense_classifier.pth"



if __name__ == "__main__":
    trainer = Trainer(
        epochs=EPOCHS,
        batch_size_load=BATCH_SIZE_LOAD,
        batch_size_train=BATCH_SIZE_TRAIN,
        cpu_count=NUM_WORKERS,
        petrinet_dir=PETRINET_DIR,
        eventlog_dir=EVENTLOG_DIR,
        output_model_path=OUTPUT_MODEL_PATH
    )
    trainer.train()
