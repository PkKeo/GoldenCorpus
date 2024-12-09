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

def is_valid_text_boundary(text):
    """
    Check if text has valid start and end characters
    Returns: bool, string with reason if invalid
    """
    if not text:
        return False, "Empty text"

    # Allow hyphen at start if it's followed by a space (dialogue)
    if text[0] in '.,!?' or (text[0] == '-' and len(text) > 1 and not text[1].isspace()):
        return False, "Invalid starting character"

    # Last character can't be hyphen
    if text[-1] == '-':
        return False, "Invalid ending character (hyphen)"

    return True, ""

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

def is_valid_text_boundary(text):
    """
    Check if text has valid start and end characters
    Returns: bool, string with reason if invalid
    """
    if not text:
        return False, "Empty text"

    # Allow hyphen at start if it's followed by a space (dialogue)
    if text[0] in '.,!?' or (text[0] == '-' and len(text) > 1 and not text[1].isspace()):
        return False, "Invalid starting character"

    # Last character can't be hyphen
    if text[-1] == '-':
        return False, "Invalid ending character (hyphen)"

    return True, ""

def is_complete_word(text, start_pos, end_pos, full_text):
    """
    Check if the text segment contains complete words
    """
    # Check if we're cutting off a word at the start
    if start_pos > 0 and full_text[start_pos-1].isalpha() and text[0].isalpha():
        return False

    # Check if we're cutting off a word at the end
    if end_pos < len(full_text) and text[-1].isalpha() and full_text[end_pos].isalpha():
        return False

    return True

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

    # Remove spaces before punctuation except after dialogue hyphen
    text = re.sub(r'(?<!-)\s+([.,!?])', r'\1', text)

    # Ensure proper spacing around hyphens except dialogue
    text = re.sub(r'(?<!^)(?<=\S)-(?=\S)', ' - ', text)

    # Convert ... to …
    text = text.replace('...', '…')

    return text

def find_valid_text_segment(text, start_pos, length, full_text):
    """
    Find valid text segments that satisfy all conditions
    Returns a list of (start, end) tuples for different length variations
    """
    max_search_distance = 50  # Maximum characters to look ahead/behind
    variations = []

    # Search forward for valid start
    start = start_pos
    while start < min(start_pos + max_search_distance, len(text)):
        # Allow hyphen if it's dialogue (followed by space)
        if text[start] in '.,!?' or (text[start] == '-' and start + 1 < len(text) and not text[start + 1].isspace()):
            start += 1
            continue

        # Try different lengths
        for l in [length - 1, length, length + 1]:  # Try original length and variations
            if l <= 0:
                continue

            end = start + l
            if end > len(text):
                continue

            # Check if segment is valid
            if is_complete_word(text[start:end], start, end, full_text):
                if text[end-1] != '-':  # Ensure it doesn't end with hyphen
                    if not variations or variations[-1] != (start, end):  # Avoid duplicates
                        variations.append((start, end))

        start += 1

    return variations if variations else [(None, None)]

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

    # Step through text with larger steps for efficiency
    step_size = 10
    for i in range(0, len(correct_text) - window_size + 1, step_size):
        # Find complete word boundaries for this window
        start_pos, end_pos = find_complete_word_boundaries(correct_text, i, window_size)
        window = correct_text[start_pos:end_pos]

        if not is_valid_text_boundary(window)[0]:
            continue

        ratio = SequenceMatcher(None, ocr_text.lower(), window.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_position = start_pos
            best_window = window

    print(f"Found start point with ratio {best_ratio:.2f}")
    print(f"Matching text: {best_window}")
    print(f"Start point search took {time.time() - start_time:.2f} seconds")
    return best_position, best_window

def find_complete_word_boundaries(text, start_pos, base_length):
    """
    Extends the selection to include complete words
    Returns start and end positions that include complete words
    """
    end_pos = start_pos + base_length

    # Extend start backwards if in middle of word
    while start_pos > 0 and text[start_pos-1].isalpha():
        start_pos -= 1

    # Extend end forwards if in middle of word
    while end_pos < len(text) and text[end_pos-1].isalpha():
        end_pos += 1

    return start_pos, end_pos

def get_word_variations(text, position, full_text):
    """
    Get variations of text by adding or removing complete words
    Returns list of (text, end_position) tuples
    """
    variations = []

    # First get the complete words in the current window
    start_pos, end_pos = find_complete_word_boundaries(full_text, position, len(text))
    complete_text = full_text[start_pos:end_pos]

    # Split into words
    words = complete_text.split()

    if len(words) < 2:  # If only one word, just return complete word
        return [(complete_text, end_pos)]

    # Try with one word less
    shorter = ' '.join(words[:-1])
    variations.append((shorter, start_pos + len(shorter)))

    # Original complete text
    variations.append((complete_text, end_pos))

    # Try adding one more word if possible
    if end_pos < len(full_text):
        next_word_match = re.search(r'^\s*\S+', full_text[end_pos:])
        if next_word_match:
            next_word = next_word_match.group()
            # Find complete boundary of next word
            _, next_end = find_complete_word_boundaries(full_text, end_pos, len(next_word))
            longer = full_text[start_pos:next_end]
            variations.append((longer, next_end))

    return variations

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

    # Get and sort text files
    text_files = sorted(
        [f for f in os.listdir(text_folder) if f.endswith('.txt')],
        key=get_page_number
    )

    # Read and combine all text files
    print("Reading and combining text files...")
    correct_text = ""
    for filename in text_files:
        with codecs.open(os.path.join(text_folder, filename), 'r', encoding='utf-8-sig') as f:
            content = process_file_content(f.read())
            # Preserve leading hyphens for dialogue
            if content.startswith('- '):
                correct_text += "\n" + content if correct_text else content
            else:
                correct_text += " " + content if correct_text else content

    correct_text = unicodedata.normalize('NFC', correct_text)

    # Initialize pointer and process rows
    correct_text_pointer = 0
    if 'correct_text' not in df.columns:
        df['correct_text'] = None

    print("\nProcessing OCR text rows...")
    for index, row in df.iterrows():
        row_start_time = time.time()
        print(f"\nProcessing row {index + 1}/{len(df)}")

        if pd.notna(row.get('correct_text')):
            print(f"Row {index + 1} already processed, skipping...")
            continue

        ocr_text = unicodedata.normalize('NFC', str(row['OCR_text']))

        if not contains_vietnamese(ocr_text):
            print(f"Row {index + 1} has no Vietnamese characters, skipping...")
            continue

        # Search for matching text
        if index == 0 or correct_text_pointer >= len(correct_text):
            position, best_match = find_start_point(ocr_text, correct_text)
            correct_text_pointer = position
        else:
            search_window = 200
            best_match = None
            best_score = float('inf')
            best_start = None

            for i in range(max(0, correct_text_pointer - search_window),
                         min(len(correct_text), correct_text_pointer + search_window)):

                # Get base window
                base_window = correct_text[i:i + len(ocr_text)]
                if not is_valid_text_boundary(base_window)[0]:
                    continue

                # Try variations with different word counts
                variations = get_word_variations(base_window, i, correct_text)

                for window, end_pos in variations:
                    if not is_valid_text_boundary(window)[0]:
                        continue

                    # Calculate score
                    score = get_edit_distance(ocr_text.lower(), window.lower())
                    position_penalty = abs(i - correct_text_pointer) / 100
                    total_score = score + position_penalty

                    if total_score < best_score:
                        best_score = total_score
                        best_match = window
                        best_start = i

            if best_start is not None:
                correct_text_pointer = best_start + len(best_match)

        if best_match:
            best_match = unicodedata.normalize('NFC', best_match)
            df.at[index, 'correct_text'] = best_match
            print(f"Found match: {best_match}")

        print(f"Row processing took {time.time() - row_start_time:.2f} seconds")

        # Save periodically
        if index % 10 == 0:
            print("Saving progress...")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Final save
    print("\nSaving final results...")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    csv_path = "input/OCR_custom 181_210.csv"
    text_folder = "text/"
    match_texts(csv_path, text_folder)
