# process.py
import os
import pandas as pd
from format import replace_text, replace_vietnamese, remove_space
from find import TextProcessor, find_ocr_position, display_match

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []

def process_files(ocr_path, text_path):
    ocr_path = os.path.join(ocr_path, 'OCR.csv')
    try:
        df = pd.read_csv(ocr_path)
        processor = TextProcessor()

        for index, row in df.iterrows():
            processor.add_line(index, row['OCR_text'])

        merged_ocr = processor.get_merged_text()
        processed_ocr = replace_text(merged_ocr)
        processed_ocr = replace_vietnamese(processed_ocr)
        processed_ocr = remove_space(processed_ocr)

        all_page_text = []
        text_files = sorted([f for f in os.listdir(text_path) if f.startswith('page') and f.endswith('.txt')],
                          key=lambda x: int(''.join(filter(str.isdigit, x))))

        for file_name in text_files:
            file_path = os.path.join(text_path, file_name)
            lines = read_text_file(file_path)
            all_page_text.extend([line.strip() for line in lines])

        merged_page = ' '.join(all_page_text)
        processed_page = replace_text(merged_page)
        processed_page = replace_vietnamese(processed_page)
        processed_page = remove_space(processed_page)

        if text_files:
            first_file = os.path.join(text_path, text_files[0])
            first_line = read_text_file(first_file)[0].strip()
            print(f"\nFirst line of original page text:")
            print(first_line)

        print(f"\nFirst line of processed OCR text:")
        processed_lines = processor.process_formatted_text(processed_ocr)
        if processed_lines:
            first_line = processed_lines.get(0, "")
            print(first_line)

        print(f"\nFirst 50 chars of processed page text:")
        print(processed_page[:50])

        position, length = find_ocr_position(processed_ocr, processed_page)
        display_match(processed_ocr, processed_page, position, length)

        return df, processed_ocr, processed_page

    except FileNotFoundError:
        print("Error: The file was not found.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None
