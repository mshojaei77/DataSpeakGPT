#OcrGPT.py

import easyocr
import g4f

def perform_ocr(image_path, lang_list):
    reader = easyocr.Reader(lang_list)
    result = reader.readtext(image_path)
    cleaned_text = ' '.join([detection[1].strip() for detection in result])
    print(f"Cleaned and joined text: {cleaned_text} \n")
    print(f"AI fixed Text: \n ")
    prompt = f"fix all words and grammar issues and rewrite this text {cleaned_text}"

    response = g4f.ChatCompletion.create(
        model= "",
        provider=g4f.Provider.GeekGpt,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for message in response:
        print(message, flush=True, end='')
    print()

if __name__ == "__main__":
    image_path = input("enter the path: ")
    supported_languages = ['en' , 'fa']
    perform_ocr(image_path, supported_languages)
