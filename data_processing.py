from pathlib import Path

import pandas as pd

DATA_PATH = Path('data')
RAW_DATA_FILE = DATA_PATH/'Ra_CNC.xlsx'
PROCESSED_DATA_FILE = DATA_PATH/'processed_ra_cnc.csv'


def process_data():

    df_raw = pd.read_excel('data/Ra_CNC.xlsx')
    df_raw.columns = ['α', 'θ', 'Vu', 'δ', 'Fx', 'Fr', 'M', 'Wz', 'Ra', 'Rz']
    df_processed_data = df_raw.drop(columns='Rz').dropna().reset_index(drop=True).round(2)
    df_processed_data.to_csv('data/processed_ra_cnc.csv', index=False)

    print(f'Missing values in raw data:\n{df_raw.isnull().sum()} \n')
    print(f'Missing values in processed data:\n{df_processed_data.isnull().sum()} \n')


if __name__ == "__main__":
    process_data()