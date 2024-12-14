import pandas as pd
import numpy as np

def process_excel_file():
    # Read the original Excel file
    input_path = 'output/OCR_with_correct.xlsx'
    output_path = 'output/OCR_with_correct_filtered.xlsx'

    # Read the Excel file
    df = pd.read_excel(input_path)

    # Store original length
    original_len = len(df)

    # Remove rows where correct_text is empty
    # This will handle both empty strings and NaN values
    df_filtered = df.dropna(subset=['correct_text']).copy()
    df_filtered = df_filtered[df_filtered['correct_text'].str.strip() != '']

    # Remove rows where Top Left is not a number
    # Convert to numeric, forcing non-numeric values to NaN
    df_filtered['Topleft'] = pd.to_numeric(df_filtered['Topleft'], errors='coerce')
    # Keep only rows where Top Left is a valid number (not NaN)
    df_filtered = df_filtered.dropna(subset=['Topleft'])

    # Save the filtered dataframe to a new Excel file
    df_filtered.to_excel(output_path, index=False)

    # Print statistics
    print(f"Original number of rows: {original_len}")
    print(f"Number of rows after filtering: {len(df_filtered)}")
    print(f"Removed {original_len - len(df_filtered)} rows total")

if __name__ == "__main__":
    process_excel_file()
