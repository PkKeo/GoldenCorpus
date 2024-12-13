import pandas as pd
import os
import glob

def process_csv_to_excel(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')

        for col in ['Topleft', 'TopRight', 'BottomLeft', 'BottomRight']:
            df[col] = df[col].astype(str)

        def process_text(text):
            if isinstance(text, str) and text.startswith('-'):
                return f"'{text}"
            return text

        df['OCR_text'] = df['OCR_text'].apply(process_text)
        df['correct_text'] = df['correct_text'].apply(process_text)

        excel_name = os.path.splitext(os.path.basename(csv_file_path))[0] + '.xlsx'
        output_file = os.path.join(os.path.dirname(csv_file_path), excel_name)

        df.to_excel(output_file, index=False)

        print(f"Converted {csv_file_path} to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")
        return None

def convert_folder_to_excel(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    processed_files = []
    for csv_file in csv_files:
        excel_file = process_csv_to_excel(csv_file)
        if excel_file:
            processed_files.append(excel_file)

    return processed_files
