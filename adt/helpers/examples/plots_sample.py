import matplotlib.pyplot as plt

# Lists to store the metrics
accuracies = []
precisions = []
recalls = []
true_positives = []
false_positives = []
parameter_sets = []

def plot_pr_curve(precisions, recalls, accuracies, parameter_sets):
    plt.figure(figsize=(10, 7))
    
    for i in range(len(precisions)):
        plt.plot(recalls[i], precisions[i], marker='o', label=f'Set {i+1}: Accuracy={accuracies[i]:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

    # Optionally, plot accuracies against parameter sets if you stored them
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accuracies)), accuracies, marker='o', label='Accuracy')
    plt.xticks(range(len(accuracies)), [f'Set {i+1}' for i in range(len(accuracies))], rotation=45)
    plt.xlabel('Parameter Sets')
    plt.ylabel('Accuracy')
    plt.title('Accuracy across Different Parameter Sets')
    plt.grid(True)
    plt.show()
