from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(predictions, ground_truth):
    # predictions: List of tuples, each containing three predicted objects (e.g., [(pred1, pred2, pred3), ...])
    # ground_truth: List of ground truth objects (e.g., [gt1, gt2, ...])
    
    # Convert predictions to binary outcomes
    binary_outcomes = []
    for preds, gt in zip(predictions, ground_truth):
        if gt in preds:
            binary_outcomes.append(1)
        else:
            binary_outcomes.append(0)
    #  binary_outcomes = [1 if gt in preds else 0 for preds, gt in zip(predictions, ground_truth)]
    binary_ground_truth = []
    
    # The ground truth for binary evaluation should be a list of 1s
    binary_ground_truth = [1] * len(ground_truth)
    
    # Calculate metrics
    accuracy = accuracy_score(binary_ground_truth, binary_outcomes)
    precision = precision_score(binary_ground_truth, binary_outcomes)
    recall = recall_score(binary_ground_truth, binary_outcomes)
    f1 = f1_score(binary_ground_truth, binary_outcomes)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

predictions = [
    ('ChoppingBoard', 'WoodenFork', 'BlackCeramicBowl'),   # Predictions for instance 1
    ('GlassCup', 'MetalSpoon', 'PlasticPlate'),            # Predictions for instance 2
    ('CeramicMug', 'Knife', 'CuttingBoard')                # Predictions for instance 3
]

ground_truth = ['ChoppingBoard', 'WoodenFork', 'BlackCeramicBowl', ]  # Ground truth objects

metrics = evaluate_predictions(predictions, ground_truth)

print(metrics)