import os
import pandas as pd

folder_path = os.path.join(os.getcwd(), 'input')

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        xlsx_file = os.path.join(folder_path, file_name)
        csv_file = os.path.join(folder_path, file_name.replace('.xlsx', '.csv'))

        data = pd.read_excel(xlsx_file)

        if 'Probability' in data.columns:
            data = data.drop(columns=['Probability'])

        data.to_csv(csv_file, index=False, encoding='utf-8-sig')
