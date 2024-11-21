from src.Trainer import Trainer
EPOCHS = 30
NUM_WORKERS = 128
BATCH_SIZE_LOAD = 1000
BATCH_SIZE_TRAIN = 50
DATA_DIR = "./data_generation/synthetic_data/"
OUTPUT_MODEL_PATH = "models/graph_sage_model_with_dense_classifier_1.pth"


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
