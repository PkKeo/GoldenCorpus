import os
import logging

logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def calculate_distance(s1, s2):
    distances = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        distances[i][0] = i
    for j in range(len(s2) + 1):
        distances[0][j] = j

    for i1, c1 in enumerate(s1, start=1):
        for i2, c2 in enumerate(s2, start=1):
            if c1 == c2:
                distances[i1][i2] = distances[i1 - 1][i2 - 1]
            else:
                distances[i1][i2] = 1 + min(distances[i1 - 1][i2 - 1],
                                            distances[i1][i2 - 1],
                                            distances[i1 - 1][i2])
    return distances[-1][-1]

def split_into_words(text):
    words = []
    current_word = ""
    for i, char in enumerate(text):
        if char == '-':
            if current_word:
                words.append(current_word)
            current_word = char
        elif char.isspace():
            if current_word:
                words.append(current_word)
                current_word = ""
        else:
            current_word += char
    if current_word:
        words.append(current_word)
    return words

def process_text_segment(text):
    replacements = {
        'f': 't', 'j': 'i', '1': 'i', '4': 'a', '7': '?',
        '...': '…', '–': '-'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()
    vietnamese_map = {
        'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'â': 'a', 'ă': 'a', 'ả': 'a', 'à': 'a',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ê': 'e',
        'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ư': 'u',
        'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ô': 'o', 'ơ': 'o', 'ỏ': 'o',
        'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
    }
    for old, new in vietnamese_map.items():
        text = text.replace(old, new)

    return text.replace(" ", "")

def is_number_only(text):
    return all(c.isdigit() or c.isspace() for c in text)

def find_best_match(ocr_line, correct_text, tracking_pos):
    if is_number_only(ocr_line):
        return "", tracking_pos

    words = split_into_words(correct_text[tracking_pos:])
    processed_ocr = process_text_segment(ocr_line)

    best_match = ""
    best_score = float('inf')
    best_end_pos = tracking_pos
    current_text = ""
    score = float('inf')

    for i in range(len(words)):

        if i == 0:
            current_text = words[i]
        else:
            current_text = current_text + " " + words[i]

        processed_correct = process_text_segment(current_text)
        last_score = score
        score = calculate_distance(processed_ocr, processed_correct)

        logging.info(f'Processed OCR: {processed_ocr}, Processed Correct: {processed_correct}, Score: {score}')

        if len(processed_correct) > len(processed_ocr) * 2:
            break

        if last_score > score:
            best_score = score
            best_match = current_text
            best_end_pos = tracking_pos + len(current_text)

    if correct_text[best_end_pos] == ' ':
        best_end_pos += 1

    return best_match, best_end_pos

def save_progress(df, output_file):
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

def create_output_csv(df, ocr_text, processed_page, correct_page, position, correct_position):
    df['correct_text'] = ''
    output_path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, 'OCR_with_correct.csv')

    if len(df) > 0:
        tracking_pos = correct_position
        for idx, row in df.iterrows():
            ocr_line = row['OCR_text'].strip()
            best_match, new_tracking_pos = find_best_match(ocr_line, correct_page, tracking_pos)
            df.at[idx, 'correct_text'] = best_match
            tracking_pos = new_tracking_pos

            if (idx + 1) % 50 == 0:
                save_progress(df, output_file)

    save_progress(df, output_file)
    return output_file
