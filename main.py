from convert import process_text_files, process_excel_files
from process import process_files
from output import create_output_csv
import os

if __name__ == "__main__":
    text_folder_path = os.path.join(os.getcwd(), 'text')
    folder_path = os.path.join(os.getcwd(), 'input')
    # process_text_files(text_folder_path)
    # process_excel_files(folder_path)
    df, processed_ocr, processed_page, correct_page, position, correct_position = process_files(folder_path, text_folder_path)
    if df is not None:
        output_file = create_output_csv(df, processed_ocr, processed_page, correct_page, position, correct_position)
        print(f"\nCreated output file: {output_file}")
