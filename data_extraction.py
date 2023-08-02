import os
import gzip
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv

class DataExtraction:
    def __init__(self):
        load_dotenv(find_dotenv())

    def extract_data(self, data_file, meta_file):
        data = []
        with gzip.open(data_file) as f:
            for l in f:
                data.append(json.loads(l.strip()))

        metadata = []
        with gzip.open(meta_file) as f:
            for l in f:
                metadata.append(json.loads(l.strip()))

        df = pd.DataFrame.from_dict(data)
        df = df[df['reviewText'].notna()]
        df_meta = pd.DataFrame.from_dict(metadata)

        return df, df_meta
