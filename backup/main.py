import pandas as pd

def process_excel_file():
    input_path = 'output/OCR_with_correct.xlsx'
    output_path = 'output/OCR_with_correct_filtered.xlsx'

    df = pd.read_excel(input_path)

    df_filtered = df.dropna(subset=['correct_text']).copy()
    df_filtered = df_filtered[df_filtered['correct_text'].str.strip() != '']

    df_filtered.to_excel(output_path, index=False)

    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after filtering: {len(df_filtered)}")
    print(f"Removed {len(df) - len(df_filtered)} rows with empty correct_text")

if __name__ == "__main__":
    process_excel_file()
