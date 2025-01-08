import os
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sentiment_analysis import get_sentiment_analysis


def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def preprocess():
    """
    Preprocess raw JSON data for Task 1 and Task 2, split into train/test sets,
    and save processed data as CSV files.
    """

     # Define paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dataset_path = os.path.join(base_path, 'dataset', '1-raw-data', 'Trainset_English.json')
    processed_data_path = os.path.join(base_path, 'dataset', '2-processed-data')
    

    # Load raw dataset
    with open(raw_dataset_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)

    # Prepare data for Task 1 and Task 2
    task1_processed_data = []
    task2_processed_data = []
    for item in raw_dataset:
        text = item.get('data', '')
        text = clean_text(text)
        promise_label = item.get('promise_status', '')
        evidence_label = item.get('evidence_status', '')
        sentiment = get_sentiment_analysis(text)
        
        task1_processed_data.append({'text':text, 'label': 1 if promise_label.lower() == 'yes' else 0, 'sentiment':sentiment})
        task2_processed_data.append({'text': text, 'label': 1 if evidence_label.lower() == 'yes' else 0, 'sentiment':sentiment})

    # Convert to DataFrames
    task1_df = pd.DataFrame(task1_processed_data)
    task2_df = pd.DataFrame(task2_processed_data)

    task1_json = task1_df.to_dict(orient='records')
    task2_json = task2_df.to_dict(orient='records')

    with open(os.path.join(processed_data_path, 'task1.json'), 'w', encoding='utf-8') as f:
        json.dump(task1_json, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(processed_data_path, 'task2.json'), 'w', encoding='utf-8') as f:
        json.dump(task2_json, f, ensure_ascii=False, indent=2)


    # Split Task 1 into train/test sets
    X_train_task1, X_test_task1, y_train_task1, y_test_task1 = train_test_split(
        task1_df['text'], task1_df['label'], test_size=0.2, random_state=42
    )
    task1_train_df = pd.DataFrame({'text': X_train_task1, 'label':y_train_task1})
    task1_test_df = pd.DataFrame({'text': X_test_task1, 'label':y_test_task1})


    # Split Task 2 into train/test sets
    X_train_task2, X_test_task2, y_train_task2, y_test_task2 = train_test_split(
        task2_df['text'], task2_df['label'], test_size=0.2, random_state=42
    )
    task2_train_df = pd.DataFrame({'text': X_train_task2, 'label':y_train_task2})
    task2_test_df = pd.DataFrame({'text': X_test_task2, 'label': y_test_task2})


    # Save DataFrames to CSV files
    task1_df.to_csv(os.path.join(processed_data_path,'task1-processed.csv'), index=False)
    task1_train_df.to_csv(os.path.join(processed_data_path, 'task1-train.csv'), index=False)
    task1_test_df.to_csv(os.path.join(processed_data_path, 'task1-test.csv'), index=False)
      
    task2_df.to_csv(os.path.join(processed_data_path,'task2-processed.csv'), index=False)
    task2_train_df.to_csv(os.path.join(processed_data_path,'task2-train.csv'), index=False)
    task2_test_df.to_csv(os.path.join(processed_data_path,'task2-test.csv'), index=False)

    print("Preprocessing and splitting completed successfully.")
