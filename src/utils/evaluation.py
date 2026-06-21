class LLMEvaluation:
    def __init__(self, llm_predictions, ground_truth):
        self.llm_predictions = llm_predictions
        self.ground_truth = ground_truth
        self.correspondences = []

        # total ground truth
        self.total_ground_truth = len(ground_truth)
        self.total_llm_activations = len(llm_predictions)
        self.total_llm_predictions = 0
        self.total_actual_correspondences = 0

        # true & false positves amd false negatives
        self.Tp = 0      # True Positives
        self.Fp = 0      # False Positives
        self.Fn = 0      # False Negatives
        self.Fp_out = 0
        self.Fp_in = 0

    def calculate_FP_FN_GT_correspondances(self):
            
        """ 
        Match LLM predictions with GT values 
        
        Cases: 
            1. LLM didnt predict for an interaction 
                - Example: GT was less than LLM time. then we check for the next GT until one is bigger than current LLM. 
                - Result: FN

            2. LLM time and prediction does not correspond to any GT interaction. (FALSE POSITIVE)
                - Example: Two consecutive LLM times are less than current gt time which means that 1st llm time does not correspond to any GT value
                - Result: FP
                
            3. LLM time and prediction corresponds to two GT values or one but we have also on FN
                - Example: Current LLM time is less than GT time but next LLM time is bigger than next GT tine which means that current LLM time corresponds to two consecutive GT predictions or to one GT with the Next GT be FN
                - Result: TP/FP double or TP/FP and 1FN
                
            4. LLM time and prediction correspomnds to one GT value
                - Example: Current LLM time is less than GT time and next LLM time is less than next GT time       
        """
    
        # Initialize variables
        llm_times = sorted(map(float, self.llm_predictions.keys()))
        gt_times  = sorted(map(float, self.ground_truth.keys()))

        # Create iterators for LLM predictions and ground truth times
        llm_iter = iter(llm_times)
        gt_iter = iter(gt_times)

        # Initialize current and next times for LLM and GT
        current_llm_time = next(llm_iter, None)
        next_llm_time = next(llm_iter, None)
        current_gt_time = next(gt_iter, None)
        next_gt_time = next(gt_iter, None)

        while current_llm_time is not None or current_gt_time is not None:
            # Case 1: Both LLM and GT times are available
            if current_llm_time is not None and current_gt_time is not None:
                if current_llm_time < current_gt_time:
                    # LLM prediction before GT time
                    # Check if next LLM prediction is also before current GT time
                    if next_llm_time is not None and next_llm_time < current_gt_time:
                        # Two consecutive LLM predictions before current GT time (False Positive)
                        self.Fp_out += 1
                        self.total_llm_predictions += 1
                        # Advance only the first LLM iterator
                        current_llm_time = next_llm_time
                        next_llm_time = next(llm_iter, None)
                        # Do not continue; allow the second LLM prediction to be considered for matching
                    else:
                        # LLM prediction corresponds to GT event
                        matched_ground_truths = []
                        time_diff = (next_gt_time - current_gt_time) if next_gt_time is not None else float('inf')

                        # Case 2: LLM prediction corresponds to two GT values
                        if next_llm_time is not None and next_gt_time is not None and next_llm_time > next_gt_time and time_diff <= 1:
                            matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                            matched_ground_truths.append((next_gt_time, self.ground_truth[next_gt_time]))
                            self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                            self.total_llm_predictions += 1
                            # Advance GT iterator twice
                            current_gt_time = next_gt_time
                            next_gt_time = next(gt_iter, None)
                            current_gt_time = next_gt_time
                            next_gt_time = next(gt_iter, None)
                        else:
                            # Case 3: LLM prediction corresponds to one GT value
                            matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                            self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                            self.total_llm_predictions += 1
                            # Advance GT iterator
                            current_gt_time = next_gt_time
                            next_gt_time = next(gt_iter, None)
                        # Advance LLM iterator
                        current_llm_time = next_llm_time
                        next_llm_time = next(llm_iter, None)
                elif current_llm_time == current_gt_time:
                    # LLM prediction matches GT time exactly
                    matched_ground_truths = [(current_gt_time, self.ground_truth[current_gt_time])]
                    self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                    self.total_llm_predictions += 1
                    # Advance both iterators
                    current_llm_time = next_llm_time
                    next_llm_time = next(llm_iter, None)
                    current_gt_time = next_gt_time
                    next_gt_time = next(gt_iter, None)
                else:
                    # GT time before LLM prediction (False Negative)
                    self.Fn += 1
                    # Advance GT iterator
                    current_gt_time = next_gt_time
                    next_gt_time = next(gt_iter, None)
            elif current_gt_time is not None:
                # LLM predictions exhausted, remaining GT times are False Negatives
                self.Fn += 1
                current_gt_time = next_gt_time
                next_gt_time = next(gt_iter, None)
            elif current_llm_time is not None:
                # GT times exhausted, remaining LLM predictions are False Positives
                self.Fp_out += 1
                self.total_llm_predictions += 1
                current_llm_time = next_llm_time
                next_llm_time = next(llm_iter, None)


    def calculate_final_TP_FP(self):
   
            """
            ********* Note *********
            
            --> in our case we predict 3 potential objects that the user may interact with.
            --> if the actual object that the user interacts with is one of the 3 potential objects it counts as TP 
                
            ********** TP **********

            --> TP : If at least 1 of the predicted objects is the GT object 
            --> FP : If none of the predicted objects is the GT objects or predictions does not correspond to any interaction
            --> TN : TN would be if the LLM predicts no interaction and no any interaction takes place. But LLM is activated for prediction so in our case there is no TN 
            --> FN : FN would be if the LLM precicts no interaction but there is intetacton. But LLM is activated for prediction so in our case there is not FN 
            """

            for llm_time, prediction, correspondence in self.correspondences:

                # Loop over the gt values that correspond to one LLM prediction
                for gt_time, gt_object in correspondence:
                    
                    self.total_actual_correspondences +=1 

                    # Flag to check if a TP is found in this correspondence
                    match_found = False  
                    
                    if gt_object in prediction:
                        self.Tp += 1
                        match_found = True
                        continue  # Break as soon as we find a match, considering it a TP
                    
                    # If no match was found, this is a false positive
                    elif not match_found:   
                        self.Fp_in +=1
            
    def calculate_metrics(self):
            
            """  
            Correspondances gives us information about the
                1. LLM total activations
                2. Actual interactions
                3. Correct predictions 
            
            Based on these three, we calculate metrics to evaluate the performance of our algorithm

            ******* Metrics *********

            1. Model_Overall_Accuracy: 
                    - Explanation: Measures the proportion of correct predictions out of total predictions
                    - Question:    Among all instances how many identied correcty as positives (relatives) and negatives (non-relatives)
                    - Intuition:   Accuracy is the most straightforward metric, measuring the overall correctness of the model. It represents the proportion of all correct predictions out of the total number of instances.
                    - Formula:     Model_Overall_Accuracy = TP + TN / TP + FP + TN + FN 
                    - Updated:     Model_Overall_Accuracy = TP / total llm activations (based on definition of Accuracy and the specifications of this problem)

            2. Precision (Positive_Prediction_Accuracy) : 
                    - Explanation: Measures the proportion of true positive predictions out of the total predicted positives.
                    - Question:    Of all the instances that were predicted as positive, how many were actually positive?   
                    - Intuition:   Precision is a metric that measures the proportion of correctly identified relevant instances out of all instances that were identified as relevant
                    - Formula:     Precision = TP / TP + FP 
            
            3. Recall (True_Positive_Rate): 
                    - Explanation: Measures the proportion of true positive predictions out of the total actual positives. 
                    - Question:    Of all the instances that were actually positive, how many were correctly predicted as positive?
                    - Intuition:   Measures the ability of a model to identify all relevant instances within a dataset. The natural intuition behind recall can be understood by considering its relationship to ”sensitivity” or ”true positive rate.
                    - Formula:     Recall = TP / TP + FN          
                    - Updated:     Recall = TP / total ground truths (since in this scenario we don't have FN or TN, we use the lenght of ground truthss
            
            4. LLM_Activation_Sensitivity 
                    - Explanation: Measures how easy the LLM is activated 
                    - Question:    From all LLM activtion how many corresponds to actual interaction
                    - Intuition:   Measures the sensitivity of the LLM to be activated 
                    - Formula:     LLM_Activation_Sensitivity = Actual_interaction / total llm activations
            
            5. LLM_Interaction_Accuracy:
                    - Explanation: Measures the proportion of correct predictions out of total actual interactions
                    - Intuition:   Measures the overall correctenss of the model. 
                    - Formula:     LLM_Interaction_Accuracy = TP / total_actual_interactions  
            """

            # Initialize counters
            self.total_ground_truth = len(self.ground_truth)            # Total number number of objects the user interacted with
            self.total_llm_activations = len(self.llm_predictions)      # Total number of LLM activated to make a predictions
            self.Fp = self.Fp_in + self.Fp_out

            # Calculate Accuracy
            self.model_overall_accuracy = round((self.Tp / (self.total_llm_predictions + self.Fn)),3) if (self.total_llm_predictions + self.Fn) else 0

            # Calculate Precision
            self.precision = round((self.Tp / (self.Tp + self.Fp)),3) if (self.Tp + self.Fp) else 0

            # Recall is always 1 in this context because there are no FN cases
            self.recall = round((self.Tp / (self.total_actual_correspondences + self.Fn)),3) if (self.total_actual_correspondences + self.Fn) else 0

            # LLM sensitivity / How sensitive is the LLM to activation 
            self.llm_activation_sensitivity = round((self.total_actual_correspondences / self.total_llm_predictions),3) if self.total_llm_predictions else 0

            # LLM correctness is close to recall but does nto take into account the FN
            self.llm_interaction_accuracy = round((self.Tp / self.total_actual_correspondences),3) if self.total_actual_correspondences else 0
            
            return (self.model_overall_accuracy, self.precision, self.recall, self.llm_activation_sensitivity, self.llm_interaction_accuracy, 
                   self.Tp, self.Fp, self.Fp_out, self.Fp_in, self.Fn, 
                   self.total_ground_truth, self.total_llm_predictions, self.total_llm_activations, self.total_actual_correspondences, 
                   self.correspondences)
    
    def display_results(self):
        # Output the correspondences 
        for llm_time, prediction, correspondence in self.correspondences:
            if isinstance(correspondence, str):
                print(f"{llm_time}: {prediction} ----------------------> {correspondence}")
            else:
                gt_descriptions = ", ".join([f"{gt_time}: '{gt_object}'" for gt_time, gt_object in correspondence])
                print(f"{llm_time}: {prediction} ----------------------> {gt_descriptions}")

        # Output the Accuracy and Recall 
        print(f"\n Correct Predictions: {self.Tp}")
        print(f"\n Non correct Predictions: {self.Fp}")
        print(f"Total Ground Truth Instances: {self.total_ground_truth}")
        # print(f"Accuracy: {self.accuracy * 100:.2f}%")
        # print(f"Precision: {self.precision * 100:.2f}%")
        # print(f"Recall: {self.recall * 100:.2f}%")
    

# Usage example:
# llm_predictions = {...}  # dictionary with LLM predictions
# ground_truth = {...}     # dictionary with ground truth data

# evaluator = LLMEvaluation(llm_predictions, ground_truth)
# correct_predictions, total_ground_truth, accuracy = evaluator.evaluate()

# evaluator.display_results()

