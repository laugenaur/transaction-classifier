import pandas as pd
import re

def clean_description(text):
    text = text.lower()
    text = text.strip() # Remove leading and trailing whitespace
    return text

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    df['description'] = (
        df['description']
        .astype(str)
        .apply(clean_description)
    )

    df.to_csv(output_file, index=False)