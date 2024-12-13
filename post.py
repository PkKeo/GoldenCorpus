import pandas as pd
import os
import glob

def process_csv_to_excel(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')

        for col in ['Topleft', 'TopRight', 'BottomLeft', 'BottomRight']:
            df[col] = df[col].astype(str)

        def process_ocr_text(text):
            if not isinstance(text, str):
                return text
            result = []
            parts = text.split()
            for part in parts:
                if part.startswith('-'):
                    result.append(f"-{part[1:]}")
                else:
                    result.append(part)
            return ' '.join(result)

        def process_correct_text(text):
            if not isinstance(text, str):
                return text
            result = []
            parts = text.split()
            for part in parts:
                if part.startswith('-'):
                    if len(part) > 1 and part[1] != ' ':
                        result.append(f"- {part[1:]}")
                    else:
                        result.append(part)
                else:
                    result.append(part)
            return ' '.join(result)

        df['OCR_text'] = df['OCR_text'].apply(process_ocr_text)
        df['correct_text'] = df['correct_text'].apply(process_correct_text)

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
