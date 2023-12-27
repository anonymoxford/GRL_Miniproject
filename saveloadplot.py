import os 
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def save_metrics(metrics, dataset_name, model_name, attempt):
    # Create directory if it does not exist
    directory = f'./metrics/{dataset_name}/{model_name}_{attempt}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, values in metrics[model_name].items():
        np_array = np.array(values)
        np.save(f'{directory}/{key}.npy', np_array)

def load_metrics(dataset_name, model_names, attempt):
    loaded_metrics = {}

    # Iterate over each model name provided in the list
    for model_name in model_names:
        # Directory where the metrics are stored for each model
        directory = f'./metrics/{dataset_name}/{model_name}_{attempt}'

        # Create a nested dictionary for each model
        loaded_metrics[model_name] = {}

        for file in os.listdir(directory):
            key = file.replace(".npy", "")  # Extracting the key from the filename
            np_array = np.load(f'{directory}/{file}')
            loaded_metrics[model_name][key] = np_array

    return loaded_metrics

def plot_metrics(dataset_name, model_names, attempt):
    loaded_metrics = load_metrics(dataset_name, model_names, attempt)
    plt.figure(figsize=(12, 6))
    # Plotting Training Loss
    plt.subplot(2, 3, 4)
    for model_name in model_names:
        plt.plot(loaded_metrics[model_name]['train_losses'], label=f'{model_name}')
    plt.title('Training Loss')
    plt.legend()

    # Plotting Test Accuracy
    plt.subplot(2, 3, 2)
    plt.axhline(y=0.75, color='black', linestyle='--')
    for model_name in model_names:
        plt.plot(loaded_metrics[model_name]['test_accuracies'], label=f'{model_name}')
    plt.title('Test Accuracy')
    plt.ylim(bottom=0, top=1)
    plt.legend()

    # Plotting Train Accuracy
    plt.subplot(2, 3, 1)
    plt.axhline(y=0.75, color='black', linestyle='--')
    for model_name in model_names:
        plt.plot(loaded_metrics[model_name]['train_accuracies'], label=f'{model_name}')
    plt.title('Train Accuracy')
    plt.ylim(bottom=0, top=1)
    plt.legend()
    
    # Plotting Val Accuracy
    plt.subplot(2, 3, 3)
    plt.axhline(y=0.75, color='black', linestyle='--')
    for model_name in model_names:
        plt.plot(loaded_metrics[model_name]['val_accuracies'], label=f'{model_name}')
    plt.title('Val Accuracy')
    plt.ylim(bottom=0, top=1)
    plt.legend()

    # Plotting Learning Rate
    plt.subplot(2, 3, 5)  # Adjust subplot positioning as needed
    for model_name in model_names:
        plt.plot(loaded_metrics[model_name]['lr'], label=f'{model_name} LR')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    os.makedirs(f'./graphs/{dataset_name}', exist_ok=True)
    plt.savefig(f'./graphs/{dataset_name}/performance_{attempt}.png')

    plt.show()
    plt.close()  # Close the figure to free memory

def print_accuracy_table(datasets, models, attempt):
    table = []

    # Header for the table
    headers = ["Model / Dataset"] + datasets

    for model_name in models:
        row = [model_name]  # First column is the model name

        for dataset_name in datasets:
            try:
                directory = f'./metrics/{dataset_name}/{model_name}_{attempt}'
                val_acc = np.load(f'{directory}/val_accuracies.npy')
                test_acc = np.load(f'{directory}/test_accuracies.npy')

                # Calculate average accuracy for each epoch
                mean_acc = (val_acc + test_acc) / 2

                # Find the best 10 epochs
                best_10_idx = np.argsort(mean_acc)[-10:]  # Indices of the best 10 epochs
                best_10_mean_acc = mean_acc[best_10_idx]

                # Calculate average of these best 10 accuracies
                final_avg_acc = np.mean(best_10_mean_acc)

                # Format the average accuracies as one entry in the row
                row.append(f'{final_avg_acc:.2%}')

            except FileNotFoundError:
                # If files don't exist, fill in with placeholders or error messages
                row.append("Data Not Found")

        table.append(row)

    # Print the table using tabulate
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
def plot_for_latex(datasets, model_names, attempt, benchmarks):
    num_datasets = len(datasets)
    plt.figure(figsize=(15, 3.5 * num_datasets))  # Adjust height based on the number of datasets

    for i, dataset_name in enumerate(datasets):
        loaded_metrics = load_metrics(dataset_name, model_names, attempt)

        # Training Loss
        plt.subplot(num_datasets, 3, i * 3 + 1)
        for model_name in model_names:
            plt.plot(loaded_metrics[model_name]['train_losses'], label=f'{model_name}')
        plt.title(f'{dataset_name} - Training Loss')
        plt.ylabel('Loss')
        if i == 0:  # Only add legend to the first plot for cleanliness
            plt.legend()

        # Training Accuracy
        plt.subplot(num_datasets, 3, i * 3 + 2)
        for model_name in model_names:
            plt.plot(loaded_metrics[model_name]['train_accuracies'], label=f'{model_name}')
        plt.title(f'{dataset_name} - Training Acc.')
        plt.ylim(bottom=0, top=1)
        plt.ylabel('Accuracy')
        

        # Test/Validation Accuracy
        plt.subplot(num_datasets, 3, i * 3 + 3)
        for model_name in model_names:
            val_acc = np.array(loaded_metrics[model_name]['val_accuracies'])
            test_acc = np.array(loaded_metrics[model_name]['test_accuracies'])
            mean_acc = (val_acc + test_acc) / 2
            plt.plot(mean_acc, label=f'{model_name}')
        plt.title(f'{dataset_name} - Mean Test/Val Acc.')
        plt.ylim(bottom=0, top=1)
        plt.axhline(y=benchmarks[dataset_name]["min"]/100, color='black', linestyle='--')
        plt.axhline(y=benchmarks[dataset_name]["max"]/100, color='black', linestyle='--')
        plt.ylabel('Accuracy')

    # Save the figure
    output_directory = f'./graphs_for_latex'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(f'{output_directory}/all_metrics_{attempt}.png', bbox_inches='tight')

    plt.close()  # Close the figure to free memory

benchmarks = {
    "Cora": {"min": 75.70, "max": 90.16},
    "CiteSeer": {"min": 64.70, "max": 82.07},
    "PubMed": {"min": 77.20, "max": 91.44},
    "ENZYMES": {"min": 53.43, "max": 78.39},
    "MUTAG": {"min": 86.44, "max": 100.00},
    "PROTEINS": {"min": 75.68, "max": 84.91}
}

datasets = ["Cora", "CiteSeer", "PubMed","ENZYMES", "MUTAG", "PROTEINS"] 
models = ["PNA", "GIN", "Att-GIN"]  
attempt = 11 

print_accuracy_table(datasets, models, attempt)
plot_for_latex(datasets, models, attempt, benchmarks)