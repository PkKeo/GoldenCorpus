# process.py
import os
import pandas as pd
from format import replace_text, replace_vietnamese, remove_space

def process_files(ocr_path, text_path):
    ocr_path = os.path.join(ocr_path, 'OCR.csv')
    try:
        df = pd.read_csv(ocr_path)

        # print("Processing OCR text line by line")
        # print("-" * 50)

        for index, row in df.iterrows():
            original_text = row['OCR_text']
            processed_text = replace_text(original_text)
            processed_text = replace_vietnamese(original_text)
            processed_text = remove_space(processed_text)

            print(f"Line {index + 1}:")
            print(f"Original : {original_text}")
            print(f"Processed: {processed_text}")
            print("-" * 50)

            df.at[index, 'OCR_text'] = processed_text

            if index > 100:
                break

        return df

    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
