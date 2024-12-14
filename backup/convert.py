import os
import codecs
import pandas as pd

def process_text_files(text_folder_path):
    for file_name in os.listdir(text_folder_path):
        if file_name.endswith('.txt'):
            txt_file = os.path.join(text_folder_path, file_name)

            with codecs.open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            with codecs.open(txt_file, 'w', encoding='utf-8') as f:
                f.write(content)

def process_excel_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            xlsx_file = os.path.join(folder_path, file_name)
            csv_file = os.path.join(folder_path, file_name.replace('.xlsx', '.csv'))

            data = pd.read_excel(xlsx_file)

            if 'Probability' in data.columns:
                data = data.drop(columns=['Probability'])

            data.to_csv(csv_file, index=False, encoding='utf-8-sig')
