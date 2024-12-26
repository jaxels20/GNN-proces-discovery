from src.Trainer import Trainer
EPOCHS = 50
NUM_WORKERS = 4
BATCH_SIZE_LOAD = 1000
BATCH_SIZE_TRAIN = 500
DATA_DIR = "./data_generation/synthetic_data/"
OUTPUT_MODEL_PATH = "models/experiment_2_model.pth"


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
