import re
text = 'vẻn vẹn-năm hào, cả tôi và Ly đều đói, Thắng bé'
text = text.strip()

# Ensure words like 'bê-tông' are split as 'bê - tông'
text = re.sub(r'(\w)-(\w)', r'\1 - \2', text)  # Insert a space around the hyphen between words
text = re.sub(r'\s+t\s+', ':', text)
text = re.sub(r'\s+t$', ':', text)
text = re.sub(r'–', '-', text)
text = re.sub(r'^\?\s+', '?', text)

text = re.sub(r'-\s+([A-Z])', r'-\1', text)
text = re.sub(r'\.\.\.', '…', text)

# Remove spaces before punctuation
text = re.sub(r'\s([.,;:?!])', r'\1', text)

# Ensure punctuation is followed by a space
text = re.sub(r'([.,;:?!])\s', r'\1 ', text)


text = text.strip()
print(text)
