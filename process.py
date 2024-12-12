# process.py
import os
import pandas as pd
from format import replace_text, replace_vietnamese, remove_space
from find import TextProcessor

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
        page_processors = {}

        text_files = sorted([f for f in os.listdir(text_path) if f.startswith('page') and f.endswith('.txt')],
                          key=lambda x: int(''.join(filter(str.isdigit, x))))

        for file_name in text_files:
            page_num = int(''.join(filter(str.isdigit, file_name)))
            file_path = os.path.join(text_path, file_name)
            lines = read_text_file(file_path)

            if page_num == 1 and lines:
                print(f"\nFirst line of original page text:")
                print(lines[0].strip())

                page_text = ' '.join([line.strip() for line in lines])
                processed_page = replace_text(page_text)
                processed_page = replace_vietnamese(processed_page)
                processed_page = remove_space(processed_page)
                print(f"\nFirst 50 chars of processed page text:")
                print(processed_page[:50])

            page_processor = TextProcessor()
            for idx, line in enumerate(lines):
                page_processor.add_line(idx, line.strip())
            page_processors[page_num] = page_processor

        for index, row in df.iterrows():
            processor.add_line(index, row['OCR_text'])

        merged_text = processor.get_merged_text()
        formatted_text = replace_text(merged_text)
        formatted_text = replace_vietnamese(formatted_text)
        formatted_text = remove_space(formatted_text)

        print(f"\nFirst line of processed OCR text:")
        processed_lines = processor.process_formatted_text(formatted_text)
        if processed_lines:
            first_line = processed_lines.get(0, "")
            print(first_line)

        return df, page_processors

    except FileNotFoundError:
        print("Error: The file was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None
