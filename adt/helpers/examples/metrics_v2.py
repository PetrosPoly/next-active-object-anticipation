# Data for LLM predictions and ground truth
llm_predictions = {
    4.999: ['Mango_A', 'ChoppingBoard', 'WoodenBowl'],
    11.831: ['BlackCeramicMug', 'Jam_Anon', 'MuffinPan'],
    18.73: ['KitchenKnife', 'WoodenSpoon', 'AirPurifier_1'],
    24.163: ['BlackCeramicMug', 'MuffinPan', 'ChoppingBoard'],
    31.362: ['Pestel', 'KitchenKnife', 'ChoppingBoard'],
    48.359: ['TvRemote_1', 'DvdRemote_1', 'BirdHouseToy'],
    61.657: ['Flask', 'Tomato_A', 'WhiteVase'],
    71.322: ['WoodenToothbrushHolder', 'CandlePattern03_1', 'DinoToy'],
    78.154: ['BlackCeramicMug', 'CoffeeCanisterLarge', 'ChoppingBoard'],
    83.853: ['KitchenKnife', 'ChoppingBoard', 'BlackCeramicMug']
}

ground_truth = {
    6.432: 'ChoppingBoard',
    19.73: 'KitchenKnife',
    20.33: 'WoodenSpoon',
    34.861: 'WoodenFork',
    48.959: 'Donut_B',
    51.458: 'Cereal_Anon',
    71.955: 'WoodenToothbrushHolder',
    73.422: 'Cracker_Anon',
    82.953: 'BlackCeramicMug'
}

# Initialize the correspondence list
correspondences = []
correct_predictions = 0
total_ground_truth = 0

# List of sorted LLM prediction times and ground truth times
llm_times = sorted(llm_predictions.keys())
gt_times = sorted(ground_truth.keys())

# Iterator for ground truth times
gt_iter = iter(gt_times)

# Get the first and next ground truth times
current_gt_time = next(gt_iter, None)
next_gt_time = next(gt_iter, None)

# Iterate through the LLM predictions
for i, llm_time in enumerate(llm_times):
    if current_gt_time is None:
        # No more ground truth interactions left
        correspondences.append((llm_time, llm_predictions[llm_time], "no actual interaction"))
        continue

    # Get the next LLM time if available
    next_llm_time = llm_times[i + 1] if i + 1 < len(llm_times) else None

    # First scenario: Two consecutive LLM times are both less than the next GT time
    if next_llm_time is not None and next_llm_time < current_gt_time:
        correspondences.append((llm_time, llm_predictions[llm_time], "no actual interaction"))
        continue

    # Matched ground truth interactions
    matched_ground_truths = []

    # Second scenario: LLM time corresponds to the current GT time
    if llm_time <= current_gt_time:
        matched_ground_truths.append((current_gt_time, ground_truth[current_gt_time]))

        # Check if the next GT time also corresponds due to being before the next LLM time
        if next_gt_time and next_llm_time and next_llm_time > next_gt_time:
            matched_ground_truths.append((next_gt_time, ground_truth[next_gt_time]))
            # Update current_gt_time to next_gt_time
            current_gt_time = next_gt_time
            next_gt_time = next(gt_iter, None)

        # Add the matched ground truths to correspondences
        correspondences.append((llm_time, llm_predictions[llm_time], matched_ground_truths))

        # Update the current GT time to the next
        current_gt_time = next_gt_time
        next_gt_time = next(gt_iter, None)

 # Reset metrics
"""  
Based on correspondances between the LLM predictions and the GT values calculate the classification metrics (Accuracy, Precision, Recall)

Note in our case we predict 3 potential objects that the user may interact with and check if there is actual interaction and if yes if the GT is one the predicted 

Assumptions: 

TP : If at least 1 of the predicted objects is the GT object 
FP : if none of the predicted objects is the GT objects or predictions does not correspond to any interaction
TN : TN would be if the LLM predicts no interaction and no any interaction takes place. But LLM is activated for prediction so in our case there is not TN 
FN : FN would be if the LLM precicts no interaction but there is intetacton. But LLM is activated for prediction so in our case there is not TN 

So we calculate the following metrics without the TN and FN 
Cases 

1. Accuracy: Measures the proportion of correct predictions out of total predictions
    
    Formula ---> Accuracy = TP + TN / TP + FP + TN + FN 
    Updated ---> Accuracy = TP / TP + FP 

2. Precision: Measures the proportion of true positive predictions out of the total predicted positives.
              Question: Of all the instances that were predicted as positive, how many were actually positive?   
              
    Formula ---> Precision = TP / TP + FP 
    
3. Recall: Measures the proportion of true positive predictions out of the total actual positives. 
           Question: Of all the instances that were actually positive, how many were correctly predicted as positive?
           
    Formula ---> Recall = TP / TP + FN          
    Updated ---> Recall = TP / total ground truths (since in this scenario we don't have FN or TN, we use the lenght of ground truthss
"""
          
# Initialize counters
total_ground_truth = len(ground_truth)        # Total number of actual interactions
total_llm_predictions = len(llm_predictions)  # Total number of predictions made by the LLM
Tp = 0                                        # True Positives
Fp = 0                                        # False Positives

for llm_time, prediction, correspondence in correspondences:

    # Case where there was no actual interaction but LLM made a prediction
    if correspondence == "no actual interaction":  
        Fp += 1  # This should count as a false positive
        continue  # Skip to the next correspondence

    # Loop over the gt values that correspond to one LLM prediction
    for gt_time, gt_object in correspondence:
        
        # Flag to check if a TP is found in this correspondence
        match_found = False  
        
        if gt_object in prediction:
            Tp += 1
            match_found = True
            continue  # Break as soon as we find a match, considering it a TP
        
        # If no match was found, this is a false positive
        elif not match_found:   
            Fp +=1

# Calculate Accuracy
accuracy = Tp / (Tp + Fp)

# Calculate Precision
precision = Tp / (Tp + Fp) 

# Recall is always 1 in this context because there are no FN cases
recall = Tp / total_ground_truth if total_ground_truth else 0

# Output the correspondences 
for llm_time, prediction, correspondence in correspondences:
    if isinstance(correspondence, str):
        print(f"{llm_time}: {prediction} ----------------------> {correspondence}")
    else:
        gt_descriptions = ", ".join([f"{gt_time}: '{gt_object}'" for gt_time, gt_object in correspondence])
        print(f"{llm_time}: {prediction} ----------------------> {gt_descriptions}")

# Output the Accuracy and Recall 
print(f"\n Correct Predictions: {Tp}")
print(f"\n Non correct Predictions: {Fp}")
print(f"Total Ground Truth Instances: {total_ground_truth}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
