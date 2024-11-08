import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from joblib import load
import numpy as np

# Đường dẫn tới mô hình tích hợp, từ điển khía cạnh và vectorizer
MODEL_PATH = "state_dict/combined_model_combined_acc_0.8435.pth"
ASPECT_DICT_PATH = "aspect_dict.txt"
NAIVE_BAYES_MODEL_PATH = "train_model/naive_bayes_model.pkl"
VECTORIZER_PATH = "train_model/tfidf_vectorizer.pkl"

# Load từ điển khía cạnh
def load_aspect_dict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        aspect_keywords = f.read().splitlines()
    return sorted(set(aspect.lower() for aspect in aspect_keywords), key=len, reverse=True)

# Load PhoBERT tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("train_model")  # Tải tokenizer từ thư mục train_model
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Tải trọng số từ file tích hợp
model.eval()

# Load Naive Bayes model
naive_bayes_model = load(NAIVE_BAYES_MODEL_PATH)

# Load TfidfVectorizer
vectorizer = load(VECTORIZER_PATH)

# Hàm nhận diện và nhóm khía cạnh với phần mô tả liên quan
def identify_aspects(sentence, aspect_dict):
    sentence = sentence.lower()
    for keyword in aspect_dict:
        if re.search(rf'\b{keyword}\b', sentence):
            return {keyword: sentence}  # Trả về khía cạnh đầu tiên tìm thấy
    return {}

# Hàm dự đoán cảm xúc với sự kết hợp PhoBERT và Naive Bayes
def predict_sentiment(sentence, aspect):
    # Bước 1: Chuẩn bị đầu vào cho PhoBERT
    inputs = tokenizer.encode_plus(
        f"Về khía cạnh {aspect}, {sentence}",
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Bước 2: Dự đoán với PhoBERT
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    phoBERT_logits = outputs.logits.numpy().flatten()

    # Tính xác suất từ logits
    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # Chuyển các xác suất thành dictionary cho các nhãn
    sentiment_scores = {
        "Negative": probabilities[0],
        "Neutral": probabilities[1],
        "Positive": probabilities[2]
    }

    # Xác định nhãn có xác suất cao nhất
    final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    return final_sentiment, sentiment_scores

# Tách câu dựa trên các dấu phẩy và các từ nối
def split_sentence(sentence):
    splitters = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ",", ";", "."]
    parts = [sentence]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return [part.strip() for part in parts if part.strip()]

# Hàm infer để trả về kết quả dự đoán
def infer(sentence):
    aspect_dict = load_aspect_dict(ASPECT_DICT_PATH)
    results = []
    
    split_words = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ";", ","]
    if any(word in sentence.lower() for word in split_words):
        sentence_parts = split_sentence(sentence.lower())
        for part in sentence_parts:
            aspects = identify_aspects(part, aspect_dict)
            if aspects:
                main_aspect = list(aspects.keys())[0]
                main_phrase = aspects[main_aspect]
                final_sentiment, sentiment_scores = predict_sentiment(main_phrase, main_aspect)
                results.append({
                    'aspect': main_aspect,
                    'phrase': main_phrase,
                    'sentiment': final_sentiment,
                    'scores': sentiment_scores
                })
    else:
        aspects = identify_aspects(sentence.lower(), aspect_dict)
        if aspects:
            main_aspect = list(aspects.keys())[0]
            main_phrase = aspects[main_aspect]
            final_sentiment, sentiment_scores = predict_sentiment(main_phrase, main_aspect)
            results.append({
                'aspect': main_aspect,
                'phrase': main_phrase,
                'sentiment': final_sentiment,
                'scores': sentiment_scores
            })
    
    return results

# Hàm chính để chạy độc lập
if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:])
    result = infer(sentence)
    print(result)
