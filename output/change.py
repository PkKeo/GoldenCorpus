import pandas as pd

def process_hyphens(text):
    if not isinstance(text, str):
        return text
    
    # Convert to list of characters for easier manipulation
    chars = list(text)
    result = []
    
    for i in range(len(chars)):
        if chars[i] == '-':
            # Handle first character
            if i == 0:
                result.append(chars[i])
                # Add space after if next character isn't already a space
                if i + 1 < len(chars) and chars[i + 1] != ' ':
                    result.append(' ')
            
            # Handle last character
            elif i == len(chars) - 1:
                # Add space before if previous character isn't already a space
                if result[-1] != ' ':
                    result.append(' ')
                result.append(chars[i])
            
            # Handle middle characters
            else:
                # Add space before if previous character isn't already a space
                if result[-1] != ' ':
                    result.append(' ')
                result.append(chars[i])
                # Add space after if next character isn't already a space
                if chars[i + 1] != ' ':
                    result.append(' ')
        
        else:
            result.append(chars[i])
    
    return ''.join(result)

# Read the Excel file
df = pd.read_excel("full1.xlsx")

# Apply the processing to the "Text" column
df['Text'] = df['Text'].apply(process_hyphens)

# Save the processed data back to Excel
df.to_excel("full.xlsx", index=False)