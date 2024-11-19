
import matplotlib.pyplot as plt

class TrainingFigureGenerator:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def create_training_figure(training_df):
        """
        Create a training figure for the training process. 
        Assumes that the training_df contains the following columns: epoch, avg_loss, time, avg_true_positives, avg_false_positives, avg_false_negatives.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Set the title and labels
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        
        # Plot the average loss
        ax.plot(training_df["epoch"], training_df["avg_loss"], label="Average Loss")
        # plot the average true positives
        ax.plot(training_df["epoch"], training_df["avg_true_positives"], label="Average True Positives")
        # plot the average false positives
        ax.plot(training_df["epoch"], training_df["avg_false_positives"], label="Average False Positives")
        # plot the average false negatives
        ax.plot(training_df["epoch"], training_df["avg_false_negatives"], label="Average False Negatives")
        # Add a legend
        ax.legend()
        
        

        return fig, ax