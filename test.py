import pandas as pd
import numpy as np



def analyze_excel(file_path):
    data = pd.read_excel(file_path)

    #filtered_data = data[data['test'] == 0] & (data['trm'] != 'OFFLINE')

    filtered_data = data[data['trm'] != 'OFFLINE']


    columns_to_anlyze = ['battV', 'battemp', 'batcur', 'acV1', 'acV2', 'acV3']

    if 'test' in data.columns:
        test_changes = (data['test'].shift(1) != data['test']) & (data['test'] == 1)
        num_tests = test_changes.sum()
        print(f"antall tester i ringen: {num_tests}")
        print(f"-" * 30)

    for col in columns_to_anlyze:
        if col in filtered_data.columns:
            column_data = filtered_data[col].dropna()
            avg = column_data.mean()
            minimum = column_data.min()
            maximum = column_data.max()
            median = column_data.median()
            std_dev = column_data.std()
            variance = column_data.var()
            skewness = column_data.skew()
            total_entries = column_data.count()


            print(f"Statistikk for {col}:")
            print(f"AVG: {avg:.2f}")
            print(f"Min: {minimum}")
            print(f"Max: {maximum}")
            print(f"Median: {median}")
            print(f"Standard dev: {std_dev:.2f}")
            print(f"varians: {variance:.2f}")
            print(f"skewness: {skewness:.2f}")

            print(f"Total entries: {total_entries}")
            print(f"-"*30)
        else:
            print(f"column {col} not found")

    if 'batcur' in filtered_data.columns:
        batcur_count_1 = (filtered_data['batcur'] == 1).sum()
        batcur_count_0 = (filtered_data['batcur'] == 0).sum()
        batcur_count_minus_1 = (filtered_data['batcur'] == -1).sum()
        print(f"Antall rader der batcur er lik 1: {batcur_count_1}")
        print(f"Antall rader der batcur er lik 0: {batcur_count_0}")
        print(f"Antall rader der batcur er lik -1: {batcur_count_minus_1}")

file_path = "AA018.xlsx"

analyze_excel(file_path)

