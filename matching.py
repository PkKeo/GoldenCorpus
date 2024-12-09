import pandas as pd
import os
import unicodedata
import re
from difflib import SequenceMatcher
import nltk
from nltk.metrics.distance import edit_distance
import time
import sys
import codecs

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def get_page_number(filename):
    """Extract page number from filename and convert to integer"""
    match = re.search(r'page(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    return 0

def preprocess_text(text, is_ocr=False):
    """
    Preprocess text according to specified rules while preserving Vietnamese characters
    """
    if not isinstance(text, str):
        text = str(text)

    # Convert newlines to spaces
    text = ' '.join(text.splitlines())

    # Convert to lowercase while preserving Vietnamese characters
    text = text.lower()

    # Normalize Unicode characters (NFC preserves Vietnamese characters)
    text = unicodedata.normalize('NFC', text)

    # Character replacements
    replacements = {
        'f': 't',
        'j': 'i',
        '1': 'i',
        '4': 'a',
        '7': '?'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove hyphens within words
    text = re.sub(r'(?<=\w)-(?=\w)', '', text)

    # Strip extra spaces while preserving Vietnamese characters
    text = ' '.join(text.split())

    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Ensure proper spacing around hyphens
    text = re.sub(r'(?<=\S)-(?=\S)', ' - ', text)

    # Convert ... to …
    text = text.replace('...', '…')

    return text

def process_file_content(content):
    """
    Process file content to convert multiline text to single line
    """
    # Replace newlines with spaces
    content = ' '.join(content.splitlines())
    # Remove extra spaces
    content = ' '.join(content.split())
    return content

def contains_vietnamese(text):
    """
    Check if text contains Vietnamese characters
    """
    vietnamese_pattern = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'
    return bool(re.search(vietnamese_pattern, text.lower()))

def get_edit_distance(str1, str2):
    """
    Calculate edit distance between two strings using dynamic programming
    """
    # Ensure both strings are properly encoded
    if not isinstance(str1, str):
        str1 = str(str1)
    if not isinstance(str2, str):
        str2 = str(str2)

    str1 = unicodedata.normalize('NFC', str1)
    str2 = unicodedata.normalize('NFC', str2)

    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                 dp[i][j-1],    # insertion
                                 dp[i-1][j-1])  # substitution
    return dp[m][n]

def find_start_point(ocr_text, correct_text):
    """
    Find the starting point in correct text that best matches the first OCR text
    """
    print(f"Finding start point for text: {ocr_text[:50]}...")
    start_time = time.time()
    best_ratio = 0
    best_position = 0
    best_window = ""
    window_size = len(ocr_text)
    total_windows = len(correct_text) - window_size + 1

    for i in range(0, total_windows, 10):  # Step size of 10 to speed up search
        window = correct_text[i:i + window_size]
        ratio = SequenceMatcher(None, ocr_text.lower(), window.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_position = i
            best_window = window

    print(f"Found start point with ratio {best_ratio:.2f}")
    print(f"Matching text: {best_window}")
    print(f"Start point search took {time.time() - start_time:.2f} seconds")
    return best_position, best_window

def is_punctuation_start(text):
    """Check if text starts with punctuation"""
    if not text:
        return False
    return text[0] in '.,!?'

def find_valid_start(text, position):
    """
    Find a valid starting position that doesn't begin with punctuation
    and doesn't split words
    """
    # If we're at a punctuation, move back to find the line start
    if is_punctuation_start(text[position:]):
        # Search backwards for the previous line's start
        while position > 0 and text[position-1] not in '.!?\n':
            position -= 1

    # Now ensure we're not in the middle of a word
    while position > 0 and text[position-1].isalpha():
        position -= 1

    return position

def match_texts(csv_path, text_folder):
    """
    Main function to match OCR text with correct text
    """
    start_time = time.time()
    print(f"Starting text matching process...")

    # Read CSV file with UTF-8 encoding
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"CSV file contains {len(df)} rows")

    # Get and sort text files numerically
    print(f"Reading text files from: {text_folder}")
    text_files = []
    for filename in os.listdir(text_folder):
        if filename.endswith('.txt'):
            text_files.append(filename)

    # Sort files by page number
    text_files.sort(key=get_page_number)

    # Read files in correct order with UTF-8 encoding
    print("Reading files in numerical order:")
    combined_text = {}
    for filename in text_files:
        page_num = get_page_number(filename)
        print(f"Processing page {page_num} ({filename})")
        with codecs.open(os.path.join(text_folder, filename), 'r', encoding='utf-8-sig') as f:
            # Process content to single line while reading
            text_content = process_file_content(f.read())
            text_content = unicodedata.normalize('NFC', text_content)
            combined_text[page_num] = text_content

    print(f"Found {len(combined_text)} text files")

    # Combine all text files in order
    print("Combining text files...")
    correct_text = ' '.join(combined_text[page] for page in sorted(combined_text.keys()))

    print(f"Total length of combined text: {len(correct_text)} characters")

    # Initialize pointers
    correct_text_pointer = 0

    # Add correct_text column if it doesn't exist
    if 'correct_text' not in df.columns:
        df['correct_text'] = None

    # Process each row in CSV
    print("\nProcessing OCR text rows...")
    for index, row in df.iterrows():
        row_start_time = time.time()
        print(f"\nProcessing row {index + 1}/{len(df)}")

        # Skip if already processed
        if pd.notna(row.get('correct_text')):
            print(f"Row {index + 1} already processed, skipping...")
            continue

        # Ensure OCR text is properly encoded
        ocr_text = str(row['OCR_text'])
        ocr_text = unicodedata.normalize('NFC', ocr_text)
        print(f"OCR text: {ocr_text}")

        # Skip if no Vietnamese characters
        if not contains_vietnamese(ocr_text):
            print(f"Row {index + 1} has no Vietnamese characters, skipping...")
            continue

        # If this is the first row or we've lost track, find start point
        if index == 0 or correct_text_pointer >= len(correct_text):
            print("Finding start point...")
            position, best_match = find_start_point(ocr_text, correct_text)
            correct_text_pointer = position

            # Use the exact match we found
            best_match = unicodedata.normalize('NFC', best_match)
            print(f"Using match: {best_match}")
            df.at[index, 'correct_text'] = best_match
            correct_text_pointer = position + len(best_match)
        else:
            # For subsequent matches, search in a smaller window around the current pointer
            print(f"Searching for match around position {correct_text_pointer}")
            best_match = None
            best_score = float('inf')
            search_window = 200  # Reduced window size

            search_start_time = time.time()
            match_count = 0

            # First try an exact match at the current pointer
            current_window = correct_text[correct_text_pointer:correct_text_pointer + len(ocr_text)]
            if current_window.lower() == ocr_text.lower():
                best_match = current_window
            else:
                # If no exact match, search in the window
                for i in range(max(0, correct_text_pointer - search_window),
                             min(len(correct_text), correct_text_pointer + search_window)):
                    # Skip if position starts with punctuation
                    if is_punctuation_start(correct_text[i:]):
                        continue

                    # Skip if we're in the middle of a word
                    if i > 0 and correct_text[i-1].isalpha() and correct_text[i].isalpha():
                        continue

                    window = correct_text[i:i + len(ocr_text)]

                    # Calculate similarity score
                    score = get_edit_distance(ocr_text.lower(), window.lower())
                    position_penalty = abs(i - correct_text_pointer) / 100  # Penalty for being far from expected position
                    total_score = score + position_penalty

                    match_count += 1

                    if total_score < best_score:
                        best_score = total_score
                        best_match = window
                        correct_text_pointer = i + len(window)

            if best_match:
                best_match = unicodedata.normalize('NFC', best_match)
                print(f"Found match with score {best_score if 'best_score' in locals() else 0}")
                print(f"OCR text: {ocr_text}")
                print(f"Best match: {best_match}")
                df.at[index, 'correct_text'] = best_match

            print(f"Compared {match_count} possible matches")
            print(f"Search took {time.time() - search_start_time:.2f} seconds")

        print(f"Row processing took {time.time() - row_start_time:.2f} seconds")

        # Save periodically with UTF-8 encoding
        if index % 10 == 0:
            print("Saving progress...")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Final save with UTF-8 encoding
    print("\nSaving final results...")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    csv_path = "input/OCR_custom 181_181.csv"
    text_folder = "text/"
    match_texts(csv_path, text_folder)
