from convert import process_text_files, process_excel_files
from process import process_files
import os

if __name__ == "__main__":
    text_folder_path = os.path.join(os.getcwd(), 'text')
    folder_path = os.path.join(os.getcwd(), 'input')
    # process_text_files(text_folder_path)
    # process_excel_files(folder_path)
    process_files(folder_path, text_folder_path)
