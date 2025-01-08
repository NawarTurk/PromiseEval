import os
import json
import pandas as pd

def generate_contingency_tables():
    languages = ['English', 'French']

    for language in languages:

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        processed_data_path = os.path.join(base_path, 'dataset', '2-processed-data')
        task1_json_path = os.path.join(processed_data_path, f'{language}_task1.json')
        task2_json_path = os.path.join(processed_data_path, f'{language}_task2.json')
        results_path = os.path.join(base_path, 'results', 'contingency_tables')  # Define results folder path
        os.makedirs(results_path, exist_ok=True)


        with open(task1_json_path, 'r', encoding='utf-8') as f:
            task1_data = json.load(f)
        
        with open(task2_json_path, 'r', encoding='utf-8') as f:
            task2_data = json.load(f)

        task1_df = pd.DataFrame(task1_data)
        task2_df = pd.DataFrame(task2_data)

        task1_contingencyTable = pd.crosstab(task1_df['label'], task1_df['sentiment'])
        task2_contingencyTable = pd.crosstab(task2_df['label'], task2_df['sentiment'])

        task1_contingencyTable_path = os.path.join(results_path, f'{language}_task1_contingency_table.csv')
        task1_contingencyTable.to_csv(task1_contingencyTable_path, index=True)
        print(f"Contingency table saved to: {task1_contingencyTable_path}")
        print(task1_contingencyTable)

        task2_contingencyTable_path = os.path.join(results_path, f'{language}_task2_contingency_table.csv')
        task2_contingencyTable.to_csv(task2_contingencyTable_path, index=True)
        print(f"Contingency table saved to: {task2_contingencyTable_path}")
        print(task2_contingencyTable)





