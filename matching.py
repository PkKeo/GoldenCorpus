import pandas as pd
import os
import unicodedata
import re
from difflib import SequenceMatcher
import time
import sys
import codecs

sys.stdout.reconfigure(encoding='utf-8')

# Global static variables
VIETNAMESE_PATTERN = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'
VIETNAMESE_COMMON_WORDS = {
    'toi', 'ban', 'anh', 'chi', 'em', 'nguoi', 'minh', 'chung', 'no', 'ho',
    'la', 'co', 'di', 'den', 've', 'noi', 'lam', 'biet', 'hieu', 'thay',
    'nghe', 'nhin', 'muon', 'can', 'phai', 'duoc', 'dang', 'se', 'da',
    'nha', 'truong', 'sach', 'ban', 'ghe', 'cua', 'xe', 'duong', 'cay',
    'hoa', 'tien', 'bach', 'viec', 'nguoi', 'nuoc', 'dat', 'troi',
    'trong', 'ngoai', 'tren', 'duoi', 'truoc', 'sau', 'giua', 'canh',
    'gan', 'xa', 'day', 'do',
    'gio', 'phut', 'giay', 'ngay', 'thang', 'nam', 'sang', 'trua',
    'chieu', 'toi', 'dem', 'som', 'muon', 'luc', 'khi',
    'ai', 'gi', 'nao', 'dau', 'sao', 'the', 'may', 'bao',
    'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
    'tram', 'nghin', 'trieu', 'nhieu', 'it',
    'tot', 'xau', 'dep', 'lon', 'nho', 'cao', 'thap', 'moi', 'cu',
    'khoe', 'yeu', 'manh', 'nong', 'lanh', 'vui', 'buon',
    'va', 'hoac', 'nhung', 'ma', 'thi', 'nen', 'vi', 'boi', 'neu',
    'khi', 'voi', 'cung', 'cho', 'de', 'den'
}

DIACRITIC_MAP = {
    'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
    'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
    'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
    'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
    'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
    'đ': 'd'
}

TEXT_REPLACEMENTS = {
    'f': 't',
    'j': 'i',
    '1': 'i',
    '4': 'a',
    '7': '?'
}

SEARCH_WINDOW_SIZE = 200
MAX_SEARCH_DISTANCE = 50
SAVE_INTERVAL = 25

def get_page_number(filename):
    """Extract page number from filename and convert to integer"""
    match = re.search(r'page(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    return 0

def is_valid_text_boundary(text):
    """Check if text has valid start and end characters"""
    if not text:
        return False, "Empty text"

    if text[0] in '.,!?' or (text[0] == '-' and len(text) > 1 and not text[1].isspace()):
        return False, "Invalid starting character"

    if text[-1] == '-' or text[-1] == '–':
        return False, "Invalid ending character (hyphen or en dash)"

    return True, ""

def preprocess_text(text, is_ocr=False):
    """Preprocess text according to specified rules while preserving Vietnamese characters"""
    if not isinstance(text, str):
        text = str(text)

    text = ' '.join(text.splitlines())
    text = text.lower()
    text = unicodedata.normalize('NFC', text)

    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)

    text = re.sub(r'(?<=\w)-(?=\w)', '', text)
    text = ' '.join(text.split())
    text = re.sub(r':\s*[-–]\s*', ': -', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'\s+[-–]\s*', r' -', text)
    text = re.sub(r'[-–]\s+', r'-', text)
    text = text.replace('...', '…')

    return text

def remove_diacritics(text):
    """Remove Vietnamese diacritics from text"""
    return ''.join(DIACRITIC_MAP.get(c, c) for c in text)

def contains_vietnamese(text):
    """Check if text contains Vietnamese characters or potential Vietnamese words"""
    if re.search(VIETNAMESE_PATTERN, text.lower()):
        return True

    words = text.lower().split()
    for word in words:
        word = re.sub(r'[.,!?]', '', word)
        if word in VIETNAMESE_COMMON_WORDS:
            return True

        word_no_diacritics = remove_diacritics(word)
        if word_no_diacritics in VIETNAMESE_COMMON_WORDS:
            return True

    return False

def process_file_content(content):
    """
    Process file content to convert multiline text to single line
    """
    content = ' '.join(content.splitlines())
    content = ' '.join(content.split())
    return content

def find_complete_word_boundaries(text, start_pos, base_length):
    """
    Extends the selection to include complete words, but stops at colon
    Returns start and end positions that include complete words
    """
    end_pos = start_pos + base_length

    while start_pos > 0 and text[start_pos-1].isalpha():
        start_pos -= 1

    colon_pos = text.find(':', start_pos, end_pos + 10)  # Look a bit ahead for colon
    if colon_pos != -1:
        return start_pos, colon_pos + 1  # Include the colon but nothing after

    while end_pos < len(text):
        if text[end_pos-1].isalpha():
            end_pos += 1
        else:
            break

    while end_pos > start_pos and (text[end_pos-1] == '–' or text[end_pos-1] == '-'):
        end_pos -= 1

    while end_pos > start_pos and text[end_pos-1].isspace():
        end_pos -= 1

    return start_pos, end_pos

def get_word_variations(text, position, full_text):
    """
    Get variations of text by adding or removing complete words
    Returns list of (text, end_position) tuples
    """
    variations = []

    start_pos, end_pos = find_complete_word_boundaries(full_text, position, len(text))
    complete_text = full_text[start_pos:end_pos]

    if complete_text.endswith(':'):
        return [(complete_text, end_pos)]

    words = complete_text.split()

    if len(words) < 2:  # If only one word, just return complete word
        return [(complete_text, end_pos)]

    shorter = ' '.join(words[:-1])
    variations.append((shorter, start_pos + len(shorter)))

    variations.append((complete_text, end_pos))

    if end_pos < len(full_text):
        next_word_match = re.search(r'^\s*\S+', full_text[end_pos:])
        if next_word_match:
            next_word = next_word_match.group()
            _, next_end = find_complete_word_boundaries(full_text, end_pos, len(next_word))
            longer = full_text[start_pos:next_end]
            variations.append((longer, next_end))

    return variations

def get_word_variations_both_ends(text, position, full_text):
    """
    Get variations of text by modifying both first and last words
    Returns list of (text, start_position, end_position) tuples
    """
    variations = []
    start_pos, end_pos = find_complete_word_boundaries(full_text, position, len(text))
    complete_text = full_text[start_pos:end_pos]

    if complete_text.endswith(':'):
        return [(complete_text, start_pos, end_pos)]

    words = complete_text.split()

    if len(words) < 3:  # Need at least 3 words to create meaningful variations
        return [(complete_text, start_pos, end_pos)]

    # Find previous word if exists
    prev_word = ''
    prev_start = start_pos
    if start_pos > 0:
        prev_match = re.search(r'\S+\s*$', full_text[:start_pos])
        if prev_match:
            prev_word = prev_match.group()
            prev_start = start_pos - len(prev_word)

    # Find next word if exists
    next_word = ''
    next_end = end_pos
    if end_pos < len(full_text):
        next_match = re.search(r'^\s*\S+', full_text[end_pos:])
        if next_match:
            next_word = next_match.group()
            next_end = end_pos + len(next_word)

    # Generate all 9 variations
    variations = [
        # Base text (keep first, keep last)
        (complete_text, start_pos, end_pos)
    ]

    # Only add variations if we have enough words
    if len(words) > 2:
        # Remove first, keep last
        text_rm_first = ' '.join(words[1:])
        variations.append((text_rm_first, start_pos + len(words[0]) + 1, end_pos))

        # Keep first, remove last
        text_rm_last = ' '.join(words[:-1])
        variations.append((text_rm_last, start_pos, start_pos + len(text_rm_last)))

    if prev_word:
        # Add prev, keep last
        text_add_prev = prev_word + ' ' + complete_text
        variations.append((text_add_prev, prev_start, end_pos))

    if next_word:
        # Keep first, add next
        text_add_next = complete_text + ' ' + next_word
        variations.append((text_add_next, start_pos, next_end))

    if len(words) > 3:
        # Remove first, remove last
        text_rm_both = ' '.join(words[1:-1])
        variations.append((text_rm_both, start_pos + len(words[0]) + 1,
                         start_pos + len(words[0]) + 1 + len(text_rm_both)))

    if prev_word and len(words) > 2:
        # Add prev, remove last
        text_add_prev_rm_last = prev_word + ' ' + ' '.join(words[:-1])
        variations.append((text_add_prev_rm_last, prev_start, prev_start + len(text_add_prev_rm_last)))

    if next_word and len(words) > 2:
        # Remove first, add next
        text_rm_first_add_next = ' '.join(words[1:]) + ' ' + next_word
        variations.append((text_rm_first_add_next, start_pos + len(words[0]) + 1, next_end))

    if prev_word and next_word:
        # Add prev, add next
        text_add_both = prev_word + ' ' + complete_text + ' ' + next_word
        variations.append((text_add_both, prev_start, next_end))

    return variations

def find_start_point(ocr_text, correct_text):
    """
    Find the starting point in correct text that best matches the first OCR text
    First finds approximate position, then fine-tunes with word variations
    """
    start_time = time.time()

    # First pass - find approximate best position
    best_ratio = 0
    best_position = 0
    window_size = len(ocr_text)

    step_size = 10
    for i in range(0, len(correct_text) - window_size + 1, step_size):
        start_pos, end_pos = find_complete_word_boundaries(correct_text, i, window_size)
        window = correct_text[start_pos:end_pos].rstrip()

        if not is_valid_text_boundary(window)[0]:
            continue

        ratio = SequenceMatcher(None, ocr_text.lower(), window.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_position = start_pos

    # Second pass - check variations only at best position
    final_best_ratio = best_ratio
    best_match = correct_text[best_position:best_position + window_size].rstrip()

    # Get variations at best position
    variations = get_word_variations_both_ends(best_match, best_position, correct_text)

    for var_text, var_start, var_end in variations:
        if not is_valid_text_boundary(var_text)[0]:
            continue

        ratio = SequenceMatcher(None, ocr_text.lower(), var_text.lower()).ratio()
        if ratio > final_best_ratio:
            final_best_ratio = ratio
            best_match = var_text
            best_position = var_start

    return best_position, best_match

def get_edit_distance(str1, str2):
    """
    Calculate edit distance between two strings using dynamic programming
    """
    str1 = preprocess_text(str1)
    str2 = preprocess_text(str2)

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
                dp[i][j] = 1 + min(dp[i-1][j],  # deletion
                                 dp[i][j-1],    # insertion
                                 dp[i-1][j-1])  # substitution
    return dp[m][n]

def match_texts(csv_path, text_folder):
    """Main function to match OCR text with correct text"""
    start_time = time.time()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    text_files = sorted(
        [f for f in os.listdir(text_folder) if f.endswith('.txt')],
        key=get_page_number
    )

    correct_text_parts = []
    for filename in text_files:
        with codecs.open(os.path.join(text_folder, filename), 'r', encoding='utf-8-sig') as f:
            content = process_file_content(f.read()).strip()
            if content:  # Only add non-empty content
                if content.startswith('- '):
                    correct_text_parts.append('\n' + content)
                else:
                    correct_text_parts.append(content)

    correct_text = ' '.join(correct_text_parts)
    correct_text = ' '.join(correct_text.split())  # Normalize whitespace
    correct_text = unicodedata.normalize('NFC', correct_text)

    correct_text_pointer = 0
    if 'correct_text' not in df.columns:
        df['correct_text'] = None

    gap_start_time = time.time()
    for index, row in df.iterrows():
        if pd.notna(row.get('correct_text')):
            continue

        ocr_text = unicodedata.normalize('NFC', str(row['OCR_text']))

        if not contains_vietnamese(ocr_text):
            continue

        if index == 0 or correct_text_pointer >= len(correct_text):
            position, best_match = find_start_point(ocr_text, correct_text)
            correct_text_pointer = position
        else:
            best_match = None
            best_score = float('inf')
            best_start = None

            for i in range(max(0, correct_text_pointer - SEARCH_WINDOW_SIZE),
                         min(len(correct_text), correct_text_pointer + SEARCH_WINDOW_SIZE)):

                base_window = correct_text[i:i + len(ocr_text)]
                if not is_valid_text_boundary(base_window)[0]:
                    continue

                variations = get_word_variations(base_window, i, correct_text)

                for window, end_pos in variations:
                    if not is_valid_text_boundary(window)[0]:
                        continue

                    score = get_edit_distance(ocr_text, window)
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

        if index % SAVE_INTERVAL == 0:
            print(f"Saving progress with {time.time() - gap_start_time:.2f} seconds")
            gap_start_time = time.time()
            df.to_csv(csv_path, index=False, encoding='utf-8-sig', lineterminator='\n')

    df.to_csv(csv_path, index=False, encoding='utf-8-sig', lineterminator='\n')
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    csv_path = "input/OCR.csv"
    text_folder = "text/"
    match_texts(csv_path, text_folder)
