#FileReaderGPT.py

import g4f
import nltk
import logging
from PyPDF2 import PdfReader
import pdfplumber
import csv
import json
import os

class FileProcessor:
    def __init__(self):
        self.token_limit = 4096
        self.gpt_response_cache = {}

    def get_file_content(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError as e:
            print(f"File not found at path: {file_path}. Error: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred while reading file: {e}")
            return ""

    def process_user_input(self):
        while True:
            try:
                path = input("Enter the file path: ")
                if path:
                    return path.strip()
                else:
                    print("Please provide a non-empty file path.")
            except Exception as e:
                print(f"An error occurred: {e}")

    def read_csv(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file)
                return "\n".join(",".join(row) for row in csv_reader)
        except Exception as e:
            print(f"An error occurred while reading CSV: {e}")
            return ""

    def read_json(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                return json.dumps(json_data, indent=2)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred while reading JSON: {e}")
            return ""

    def read_pdf(self, file_path):
        try:
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                return text
        except Exception as e:
            print(f"An error occurred while reading PDF with PyPDF2: {e}")

            # Trying alternative PDF extraction library (pdfplumber)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    return text
            except Exception as e:
                print(f"An error occurred while reading PDF with pdfplumber: {e}")
                return ""

    def read_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as txt_file:
                return txt_file.read()
        except Exception as e:
            print(f"An error occurred while reading TXT: {e}")
            return ""

    def chunk_text(self, prompt):
        sentences = nltk.sent_tokenize(prompt)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.token_limit:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_and_cache_chunk(self, chunk):
        if chunk in self.gpt_response_cache:
            print("\nCached GPT Response:\n", self.gpt_response_cache[chunk])
            logging.info("GPT response fetched from cache.")
        else:
            stream = g4f.ChatCompletion.create(
                model=g4f.models.default,
                provider=g4f.Provider.GeekGpt,
                messages=[
                    {"role": "user", "content": chunk}
                ],
                stream=True
            )

            response_content = ""
            for response_chunk in stream:
                if response_chunk:
                    content = response_chunk
                    if content is not None:
                        response_content += content

            self.gpt_response_cache[chunk] = response_content
            print(response_content)

    def validate_file(self, file_path):
        valid_extensions = (".csv", ".json", ".pdf", ".txt")
        if not file_path.lower().endswith(valid_extensions):
            print(f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
            return False

        # Additional file content validation based on type
        if file_path.lower().endswith((".csv", ".json", ".txt")):
            content = self.get_file_content(file_path)
            if not content.strip():
                print("The file is empty or could not be read. Please provide a valid file with content.")
                return False

        return True

    def main(self):
        try:
            file_path = self.process_user_input()

            if not file_path:
                print("File path is empty. Exiting.")
                return

            if not os.path.isfile(file_path):
                print(f"File not found at path: {file_path}")
                return

            if not self.validate_file(file_path):
                return

            if file_path.lower().endswith((".csv", ".json", ".pdf", ".txt")):
                if file_path.lower().endswith(".csv"):
                    text = self.read_csv(file_path)
                elif file_path.lower().endswith(".json"):
                    text = self.read_json(file_path)
                elif file_path.lower().endswith(".pdf"):
                    text = self.read_pdf(file_path)
                elif file_path.lower().endswith(".txt"):
                    text = self.read_txt(file_path)
            else:
                text = self.get_file_content(file_path)

            if not isinstance(text, str):
                print("File content could not be read as a string. Please provide a valid file with readable content.")
                return

            if len(text.strip()) == 0:
                print("The file is empty or could not be read. Please provide a valid file with content.")
                return

            prompt = text

            if len(prompt) > self.token_limit:
                print("Warning: Text length exceeds token limit. Processing in chunks.")
                chunks = self.chunk_text(prompt)
            else:
                chunks = [prompt]

            for chunk in chunks:
                self.process_and_cache_chunk(chunk)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    FileProcessor().main()
