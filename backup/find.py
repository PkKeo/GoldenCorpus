class TextProcessor:
    def __init__(self):
        self.original_lines = {}
        self.merged_text = ""
        self.line_mappings = {}
        self.current_position = 0

    def add_line(self, index, text):
        self.original_lines[index] = text
        if self.merged_text:
            self.merged_text += " "
            self.current_position += 1
        start_pos = self.current_position
        self.merged_text += text
        end_pos = self.current_position + len(text)
        self.line_mappings[index] = (start_pos, end_pos)
        self.current_position = end_pos

    def get_merged_text(self):
        return self.merged_text

    def get_original_line(self, index):
        return self.original_lines.get(index)

    def get_line_position(self, index):
        return self.line_mappings.get(index)

    def process_formatted_text(self, formatted_text):
        processed_lines = {}
        for index, (start, end) in self.line_mappings.items():
            if start < len(formatted_text):
                actual_end = min(end, len(formatted_text))
                processed_lines[index] = formatted_text[start:actual_end]
        return processed_lines

def find_ocr_position(ocr_text, page_text):
    search_text = ocr_text[:50]
    best_position = 0
    best_matches = 0

    for start in range(len(page_text)):
        matches = 0
        for i in range(min(len(search_text), len(page_text) - start)):
            if search_text[i] == page_text[start + i]:
                matches += 1

        if matches > best_matches:
            best_matches = matches
            best_position = start

    best_length = min(len(search_text), len(page_text) - best_position)
    return best_position, best_length

def map_position(processed_text, correct_text, processed_pos):
    p_idx = c_idx = 0
    while p_idx < processed_pos and p_idx < len(processed_text) and c_idx < len(correct_text):
        if processed_text[p_idx].isspace():
            p_idx += 1
            continue
        if correct_text[c_idx].isspace():
            c_idx += 1
            continue
        p_idx += 1
        c_idx += 1
    return c_idx

def display_match(ocr_text, page_text, position, length):
    print("\nMatch Results:")
    print("OCR:", ocr_text[:50])
    print("PAG:", page_text[position:position+length])
