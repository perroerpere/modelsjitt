import pandas as pd
import numpy as np



def analyze_excel(file_path):
    data = pd.read_excel(file_path)

    filtered_data = data[data['test'] == 0]


    columns_to_anlyze = ['battV', 'battemp', 'batcur', 'acV1', 'acV2', 'acV3']

    for col in columns_to_anlyze:
        if col in data.columns:
            column_data = data[col].dropna()
            avg = column_data.mean()
            minimum = column_data.min()
            maximum = column_data.max()
            median = column_data.median()
            total_entries = column_data.count()

            print(f"Statistikk for {col}:")
            print(f"AVG: {avg:.2f}")
            print(f"Min: {minimum}")
            print(f"Max: {maximum}")
            print(f"Median: {median}")
            print(f"Total entries: {total_entries}")
            print(f"-"*30)
        else:
            print(f"column {col} not found")

    if 'test' in data.columns:
        test_changes = (data['test'].shift(1) != data['test']) & (data['test'] == 1)
        num_tests = test_changes.sum()
        print(f"antall tester i ringen: {num_tests}")

file_path = "AA008.xlsx"

analyze_excel(file_path)

