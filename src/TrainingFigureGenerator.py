import seaborn as sns
import matplotlib.pyplot as plt

class TrainingFigureGenerator:
    
    def __init__(self) -> None:
        pass
    
    def create_training_figure(training_df):
        """
        Create a visually enhanced training figure for the training process. 
        Assumes that the training_df contains the following columns: epoch, avg_loss, avg_true_positives, avg_false_positives, avg_false_negatives.
        """
        # Apply a professional-looking style
        import seaborn as sns
        sns.set_context("paper")
        sns.set_style("whitegrid")

        # Create a figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)

        # Plot the data with distinct line styles and markers
        ax.plot(
            training_df["epoch"], training_df["avg_loss"], 
            label="Average Loss", color="blue", linestyle="-", linewidth=2, marker="o"
        )
        ax.plot(
            training_df["epoch"], training_df["avg_true_positives"], 
            label="Average True Positives", color="green", linestyle="--", linewidth=2, marker="s"
        )
        ax.plot(
            training_df["epoch"], training_df["avg_false_positives"], 
            label="Average False Positives", color="orange", linestyle=":", linewidth=2, marker="^"
        )
        ax.plot(
            training_df["epoch"], training_df["avg_false_negatives"], 
            label="Average False Negatives", color="red", linestyle="-.", linewidth=2, marker="d"
        )

        # Set the title and labels
        ax.set_title("Training Metrics Over Epochs", fontsize=16, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=14)

        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add a legend with a professional look
        ax.legend(fontsize=12, title="Metrics", title_fontsize=12, loc="upper right", frameon=True)

        # Add gridlines for clarity
        ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Optimize layout
        plt.tight_layout()

        return fig, ax