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
