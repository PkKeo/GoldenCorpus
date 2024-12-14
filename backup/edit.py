import pandas as pd
import os

def reverse_coordinates_by_page(input_file, output_folder):
    try:
        df = pd.read_excel(input_file)

        required_columns = ['Page', 'tl', 'tr', 'bl', 'br', 'OCR', 'text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in the Excel file")

        df_reversed = df.copy()

        unique_pages = df['Page'].unique()

        for page in unique_pages:
            page_mask = df['Page'] == page

            coords = df.loc[page_mask, ['tl', 'tr', 'bl', 'br']].values

            coords_reversed = coords[::-1]

            df_reversed.loc[page_mask, ['tl', 'tr', 'bl', 'br']] = coords_reversed

        output_filename = f"OCR_reversed.xlsx"
        output_path = os.path.join(output_folder, output_filename)

        df_reversed.to_excel(output_path, index=False)

        print(f"Ngon")
        return output_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = os.path.join("output", "OCR.xlsx")
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    output_file = reverse_coordinates_by_page(input_file, output_folder)

    if output_file:
        print(f"File processed successfully. Output saved to: {output_file}")
    else:
        print("Failed to process the file.")
