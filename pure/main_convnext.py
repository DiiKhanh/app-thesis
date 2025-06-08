
import sys
import os
# Lấy thư mục chứa file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Truy ngược lên các thư mục cha để thêm đường dẫn gốc chứa helper_code
root_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_path)

print(root_path)
from team_code_convnext import *
import time
from helper_code import find_data_folders
from sklearn.model_selection import train_test_split



# Define the data and model folder
data_folder = "../../../../training"
model_folder = "models/convnext"


# split train/ test
patient_ids = find_data_folders(data_folder)
train_ids, test_ids = train_test_split(patient_ids,test_size=0.2, random_state=42)
print('Number of test patients: ', len(test_ids))

 
# ### TRAIN MODEL VIT


print("Start train model")
start_time_train_model = time.time()
verbose = 1

train_challenge_model(data_folder, model_folder, train_ids, verbose) 

print(f"Finished running train_model.py in {round((time.time() - start_time_train_model)/60, 2)} minutes.")


plot_metrics_from_csv(os.path.join(model_folder, 'eeg', TRAIN_LOG_FILE_NAME))


 
### RUN MODEL VIT


import pandas as pd
import numpy as np, scipy as sp, os, sys
from tqdm import tqdm
import time
from helper_code import *
from team_code_convnext import load_challenge_models, run_challenge_models


output_filename = 'convnext-001.csv'
output_folder = 'results'

def run_model(model_folder, data_folder, output_folder,patient_ids,  allow_failures, verbose):
    if verbose >= 1:
        print('Loading the Challenge models...')

    
    models = load_challenge_models(model_folder, verbose) 

    # Find the Challenge data.
    if verbose >= 1:
        print('Finding the Challenge data...')

    num_patients = len(patient_ids)

    if num_patients==0:
        raise Exception('No data were provided.')


    os.makedirs(output_folder, exist_ok=True)


    if verbose >= 1:
        print('Running the Challenge models on the Challenge data...')
    prediction_results = []
    for i in tqdm(range(num_patients)):
        print(f"[{i}/{num_patients}]")
       

        patient_id = patient_ids[i]
        print('PATIENT ID: ', patient_id )
        file_meta_data = os.path.join(data_folder, patient_id, f"{patient_id}.txt")
        meta_data = load_text_file(file_meta_data)
        actual_outcome = get_variable(meta_data, 'Outcome', str)
        actual_cpc = get_cpc(meta_data)

        # Allow or disallow the model(s) to fail on parts of the data; this can be helpful for debugging.
        try:
            outcome_binary, outcome_probability = run_challenge_models(models, data_folder, patient_id, verbose, False) ### Teams: Implement this function!!!
            print("Predict outcome: ", outcome_binary, " - Actual outcome: ", actual_outcome)
           
            prediction_results.append({
                'patient_id': patient_id,
                'outcome_binary': outcome_binary,
                'outcome_prob': outcome_probability, 
                'actual_outcome': actual_outcome
            })
            
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                outcome_binary, outcome_probability= float('nan'), float('nan')
            else:
                raise

        #Save output to csv
        
    save_predictions_to_csv(prediction_results, output_folder, output_filename)

    if verbose >= 1:
        print('Done.')

# Call to Run model here

print("Start run model")
start_time_run_model = time.time()

# Allow or disallow the model to fail on parts of the data; helpful for debugging.
allow_failures = False

verbose = 1

run_model(model_folder, data_folder, output_folder, test_ids, allow_failures, verbose)

print(f"Finished running run_model.py in {round((time.time() - start_time_run_model)/60, 2)} minutes.")

 
# ### Evaluate model

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from helper_code import load_predictions_from_csv


all_results = load_predictions_from_csv(output_folder, output_filename)
y_pred = list()
y_true = list()
y_prob = list()
for res in all_results:
    y_pred.append(res['outcome_binary'])
    y_true.append(res['actual_outcome'])
    y_prob.append(res['outcome_prob'])


acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)



print("=== Evaluation Metrics ===")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1 Score       : {f1:.4f}")

save_metrics_to_csv(os.path.join(model_folder, 'eeg', "metrics_on_test.csv"), acc, prec, rec, f1)



def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()  
    plt.savefig(filename)  
    plt.close()  
    
    
# Gọi hàm và lưu file
plot_confusion_matrix(y_true, y_pred, filename=os.path.join(model_folder, 'eeg', "confusion_matrix.png"))


