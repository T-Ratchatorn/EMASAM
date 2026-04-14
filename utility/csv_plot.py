import os
import pandas as pd
import matplotlib.pyplot as plt


def csv_plot(csv_path, column_x_axis, column_y_axis, plot_name):
    """
    Reads a CSV file and creates a 2D line plot of the specified columns,
    then saves the figure to the same folder as the CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        column_x_axis (str): Column name for the x-axis.
        column_y_axis (str): Column name for the y-axis.
        plot_name (str): Name for the saved plot (without extension).
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    df[column_x_axis] = pd.to_numeric(df[column_x_axis], errors="coerce")
    df[column_y_axis] = pd.to_numeric(df[column_y_axis], errors="coerce")
    df = df.dropna(subset=[column_x_axis])
    df = df.dropna(subset=[column_y_axis])
    
    # Create the plot
    plt.figure()
    plt.plot(df[column_x_axis], df[column_y_axis], label=plot_name)
    
    plt.xlabel(column_x_axis)
    plt.ylabel(column_y_axis)

    # Save the figure to the same folder
    folder = os.path.dirname(csv_path)
    save_path = os.path.join(folder, f"{plot_name}.png")
    plt.savefig(save_path)
    plt.close()
