import pandas as pd
import os

def create_output_csv(df, ocr_text, processed_page, correct_page, position, correct_position):
    df['correct_text'] = ''

    if len(df) > 0:
        current_pos = correct_position
        for idx, row in df.iterrows():
            ocr_length = len(row['OCR_text'])
            processed_length = len(row['OCR_text'].strip())
            if current_pos + processed_length <= len(correct_page):
                df.at[idx, 'correct_text'] = correct_page[current_pos:current_pos + processed_length]
            current_pos += processed_length + 1

    output_path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'OCR_with_correct.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    return output_file
