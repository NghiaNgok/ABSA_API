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
OVERALL_MODEL_PATH = "state_dict/phobert_sentiment_model_val_acc_0.8053.pth"

# Load từ điển khía cạnh
def load_aspect_dict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        aspect_keywords = f.read().splitlines()
    return sorted(set(aspect.lower() for aspect in aspect_keywords), key=len, reverse=True)
# Danh mục khía cạnh
aspect_categories = {
    "Chất lượng sản phẩm": [
        "chất lượng", "hiệu năng", "độ bền", "màn hình", "độ phân giải", "âm thanh", "loa", "chất lượng màn hình",
        "hiển thị", "hình ảnh", "cảm biến", "sức mạnh", "chất lượng hiển thị", "phần mềm", "camera", "máy ảnh", "máy"
        "đa nhiệm", "chụp ảnh", "cảm biến vân tay", "nét chữ", "kết nối wifi", "độ sáng", "ram", "hiệu quả", "tiếng chuông",
        "bắt wifi", "sóng", "tín hiệu", "độ tương phản màn hình", "phát wifi", "màn hình tần số quét", "độ hoàn thiện",
        "tốc độ", "xử lý", "tốc độ phản hồi", "sản phẩm", "phục vụ", "chất luọng", "chât lương", "sản pham",
        "chất lượng đt", "chất lượng máy", "chất lượng sản phẩm", "độ hoàn thiện", "chất lượng giá", "chat lượng",
        "điểm chết trên màn hình", "màu sắc hình ảnh", "độ phân giải màn hình", "hình ảnh chữ", "PC", "Laptop", "Máy tính bảng", "Máy tính", "Máy tính xách tay", "Máy tính cầm tay",
        # New terms
        "chất lượng sp", "chất vải", "chất", "chất áo", "chất liệu vải", "chất liệu", "chất da", "điện thoại", 
        "thân máy", "đt", "may", "điên thoai", "dien thoai", "đồ nokia", "đt nokia", "iphone", "iphone 13", "thiết bị", "áo", "quần", "vải", "vải áo", "vải quần", "ống quần", "kích cỡ", "form áo", "cổ áo", "áo polo", "áo mút"
        ,"set áo", "set đồ", "áo croptop", "khuy", "khuy áo", "balo","ốp lưng","màu sắc màn hình", "giày", "giầy", "dép","ứng dụng camera", "hàng hóa", "san phẩm", "san pham", "sp", "hàng hoá", "sản phảm"
        , "hàng", "pin"
    ],
    "Giá cả và chi phí": [
        "giá", "giá cả", "giá tiền", "tầm giá", "chi phí", "túi tiền", "mức giá", "giá thành", "gia", "giá cã", 
        "gía", "tiền"
    ],
    "Thiết kế và hình thức": [
        "thiết kế", "màu sắc", "kích thước", "vỏ máy", "hình thức", "vỏ kim loại", "vỏ hộp", "kiểu dáng", "ngoại hình",
        "màu vàng", "màu đen", "màu trắng", "màu xanh", "màu đỏ", "khung viền", "viền màn hình", "cầm nắm", "trọng lượng",
        "vỏ nhôm", "tông màu", "hộp ngoài", "mặt lưng bóng", "chống sốc", "mỏng", "điểm chết trên màn hình",
        "mẫu", "mẫu mã", "màn cạnh bên trái", "góc nhìn", "góc trái màn hình", "bên ngoài", "bên phải",
        # New terms
        "màu", "size",
        "chiều rộng", "chiều dài", "nắp túi", "đường chỉ", "dây quay cắp", "đường may", "quần lót", "túi",
        "cúc áo", "da", "kiểu dáng", "chữ in", "vải thun", "váy", "áo sơ mi", "ống tay", "vai", 
        "tay áo", "chân váy", "chữ", "đường kim mũi"
        
    ],
    "Trải nghiệm sử dụng": [
        "cảm giác", "cảm ứng", "sử dụng", "bàn phím", "chơi game", "trải nghiệm", "nghe gọi", "chơi liên quân", "đọc sách",
        "lướt web", "lấy nét", "âm thanh cuộc gọi", "cảm nhận đầu tiên", "nghe nhạc", "xài app", "mọi người", "mọi chức năng",
        "mọi ứng dụng", "thao tác", "dùng", "sử dụng", "khởi động", "hoạt động", "tác vụ", "ứng dụng", "làm việc",
        "phần cứng", "dữ liệu lưu sdt",
        # New terms
        "mặc",  "mùi", "sách"
    ],
    "Dịch vụ và hỗ trợ khách hàng": [
        "dịch vụ", "hỗ trợ", "chăm sóc khách hàng", "nhân viên hỗ trợ", "nv giao hàng", "dịch vụ ship", "thái độ",
        "hỗ trợ kỹ thuật", "support", "nhân viên cskh", "phản hồi thông tin", "nhân viên chăm sóc khách hàng", "cskh",
        "dịch vụ giao hàng", "dịch vụ hỗ trợ", "dịch vụ hỗ trợ từ nhà cung cấp", "thái độ phục vụ", "tư vấn", "shop", "shop xử lý",
        "người bán", "bán hàng", "nhân viên", "nv giao hàng", "đội ngũ tiki", "nhân viện hỗ trợ", "nhân viên giao hàng",
        "chăm sóc", "chăm sóc khách", "hàng giao", "tư vấn", "chăm sóc khách hàng", "trách nhiệm",
        # New terms
        "phục vụ", "tư vấn", "chất lượng dịch vụ"
    ],
    "Vận chuyển và đóng gói": [
        "đóng gói", "vận chuyển", "thời gian giao hàng", "ship", "đơn vị vận chuyển", "vận chuyển đóng gói", "cách đóng gói",
        "bao bì", "bao gói", "gói quà", "đóng hộp", "ship hàng", "nhận hàng đóng gói", "đóng ngói", "nhận hành",
        "giao hàng", "nhân viên giao hàng", "đóng giói", "đóng hangd", "vận chuyện", "dong hàng", "giao hang",
        "cách đóng hàng", "đóng gói sản phẩm", "hộp đóng gói", "hộp giao", "đóng hàng", "khâu gói hàng", "khâu vận chuyển",
        "vỏ hộp", "vận chuyện", "giao hành", "hộp ngoài", "vận chuyển đóng gói", "bọc gói", "anh vận chuyển", 
        "bao", "đóng gói hàng", "phần cứng", "bưu tá", "kiện hàng", "shiper", "đường truyền",
        # New terms
        "đóng gói sản phẩm", "đóng gói", "hộp carton", "hộp đựng giày", "hộp đựng giầy"
    ],
    "Phụ kiện và tính năng bổ sung": [
        "phụ kiện", "tai nghe", "bao da", "cảm biến vân tay", "dây sạc", "sạc pin", "dây gắn tai nghe",
        "cổng sạc", "mic", "ram", "cục pin", "loa ngoài", "loa trong", "đèn vàng", "chữ to", "thời lượng pin", "case tặng kèm",
        "túi tặng kèm", "camera kép", "sản phẩm khuyến mại", "phần mềm", "máy đọc sách", "máy tính bảng", "ipad",
        "túi chống sốc", "take note", "dán seal", "bóc seal", "miếng dán màn hình", "cổng sạc c", "bàn phím bấm",
        "dark mode", "hộp sạc", "pin dung lượng", "âm lượng phát", "cục pin", "dung lượng", "bản mới", "camera trước",
        "camera chụp ảnh", "đèn", "phiếu bảo hành", "dây", "hộp điện thoại", "túi bảo vệ", "nghe nói", "micro máy", 
        "thân máy và sạc", "mắt", "kính", "lỗ sạc", "củ sạc", "gói",
        # New terms
        "âm thanh", "dây sạc", "sáp khử mùi", "sáp", "khóa", "ví", "mực in", "sáp", "đừơg may"
    ],
    "Hóa đơn và giấy tờ": [
        "giấy tờ", "hóa đơn", "xuất hoá đơn", "giấy tờ bảo hành", "hoá đơn bán hàng", "hoá đơn vat", "xuất hóa đơn",
        "hóa đơn gtgt", "mã imei", "chính sách bảo hành", "thẻ bảo hành", "chính sách đổi trả", "trả hàng", "giấy bảo hành",
        "hoá đơn điện tử", "giấy bảo hành", "phiếu bảo hành", "xuất hoá đơn vat",
        # New terms
        "giấy", "thời gian bảo hành"
    ],
    "Khác": [
        "quà tặng", "đổi", "ưu đãi", "hộp", "hàng dùng thử", "bản kid", "hàng điện tử",
        "gói hàng", "hàng nhập khẩu", "hàng ref", "mọi thứ", 
        "phiên bản", "bản thường", "cập nhật", "bản kid", "đơn hàng", "cập nhật", "portrait", "đổi","thấm hút"
        # New terms
        "công nghệ", "gối", "đồ", "kem", "son", "tất", "nội dung"
    ]
}

# Hàm phân loại từ vào danh mục với kiểm tra từ khóa chính xác
def classify_aspect(word):
    for category, keywords in aspect_categories.items():
        # Sử dụng regex để chỉ tìm từ khóa chính xác
        if any(re.search(rf'\b{re.escape(keyword)}\b', word) for keyword in keywords):
            return category
    return "Khác"

# Load PhoBERT tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("train_model") 
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load Naive Bayes model và Vectorizer
naive_bayes_model = load(NAIVE_BAYES_MODEL_PATH)
vectorizer = load(VECTORIZER_PATH)

# Load overall sentiment model
overall_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
overall_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
overall_model.load_state_dict(torch.load(OVERALL_MODEL_PATH, map_location=torch.device('cpu')))
overall_model.eval()

# Hàm nhận diện và nhóm khía cạnh với phần mô tả liên quan
def identify_aspects(sentence, aspect_dict):
    sentence = sentence.lower()
    for keyword in aspect_dict:
        if re.search(rf'\b{keyword}\b', sentence):
            return {keyword: sentence}
    return {}

# Hàm dự đoán cảm xúc
def predict_sentiment(sentence, aspect):
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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    sentiment_scores = {
        "Positive": probabilities[2],
        "Neutral": probabilities[1],
        "Negative": probabilities[0]
    }
    final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    return final_sentiment, sentiment_scores

# Hàm dự đoán overall sentiment cùng với scores
def predict_overall_sentiment(sentence):
    inputs = overall_tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = overall_model(input_ids=input_ids, attention_mask=attention_mask)
    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    overall_scores = {
        "Positive": probabilities[2],
        "Neutral": probabilities[1],
        "Negative": probabilities[0]
    }
    overall_sentiment = max(overall_scores, key=overall_scores.get)
    
    return overall_sentiment, overall_scores

# Tách câu dựa trên các dấu câu
def split_sentence(sentence):
    splitters = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ",", ";", "."]
    parts = [sentence]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return [part.strip() for part in parts if part.strip()]

# Tách câu dựa trên các dấu câu và từ "và", "cộng" nhưng xử lý linh hoạt cho khía cạnh
def split_sentence(sentence):
    # Tách câu dựa trên các từ nối phổ biến và "và", "cộng"
    splitters = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ",", ";", ".", " và ", " cộng "]
    parts = [sentence]
    
    # Tách các phần câu dựa trên các từ nối
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    
    return [part.strip() for part in parts if part.strip()]
# Hàm tách phần trước từ "của"
def extract_aspect_before_cua(sentence):
    # Tìm từ "của" trong câu
    match = re.search(r"(.*?)\s+của\s+", sentence)  # Tìm phần trước từ "của"
    if match:
        return match.group(1).strip()  # Trả về phần trước "của"
    else:
        return sentence  # Nếu không có "của", trả về toàn bộ câu
def infer(sentence):
    aspect_dict = load_aspect_dict(ASPECT_DICT_PATH)
    results = []
    identified_aspects = set()  # Track identified aspects to avoid duplicates
    
    # Dự đoán overall sentiment của toàn bộ câu và scores
    overall_sentiment, overall_scores = predict_overall_sentiment(sentence)
    
    # Tách câu dựa trên các từ nối phổ biến
    split_words = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ";", ",", ".", " và ", " cộng "]
    if any(word in sentence.lower() for word in split_words):
        sentence_parts = split_sentence(sentence.lower())
        
        # Lưu trữ khía cạnh và cảm xúc của phần trước nếu không có cảm xúc rõ ràng
        previous_aspect = None
        previous_sentiment = None
        previous_scores = None
        reference_phrases = ["như thế", "như vậy", "cũng vậy", "cũng như thế", "giống vậy"]  # Các cụm từ để nhận diện tham chiếu cảm xúc trước đó
        
        for part in sentence_parts:
            # Kiểm tra và xử lý trường hợp có "của"
            part_before_cua = extract_aspect_before_cua(part)  # Tách phần trước từ "của"
            aspects = identify_aspects(part_before_cua, aspect_dict) if part_before_cua != part else identify_aspects(part, aspect_dict)
            
            if aspects:
                # Xác định khía cạnh và phân loại câu
                main_aspect = list(aspects.keys())[0]
                main_phrase = aspects[main_aspect]
                
                # Skip if aspect already identified
                if main_aspect in identified_aspects:
                    continue
                
                category = classify_aspect(main_aspect)
                final_sentiment, sentiment_scores = predict_sentiment(main_phrase, main_aspect)
                
                # Kiểm tra nếu khía cạnh có các cụm từ "như thế", "như vậy" và không có cảm xúc rõ ràng
                if any(ref in main_phrase for ref in reference_phrases) and previous_sentiment:
                    # Sử dụng cảm xúc của khía cạnh trước đó
                    final_sentiment = previous_sentiment
                    sentiment_scores = previous_scores
                
                # Cập nhật khía cạnh và cảm xúc cho phần hiện tại
                previous_aspect = main_aspect
                previous_sentiment = final_sentiment
                previous_scores = sentiment_scores
                
                # Mark aspect as identified to avoid duplicates
                identified_aspects.add(main_aspect)
            
                # Thêm vào kết quả
                results.append({
                    'overall_sentiment': overall_sentiment,
                    'overall_scores': overall_scores,
                    'phrase': main_phrase,
                    'aspect': main_aspect,
                    'sentiment': final_sentiment,
                    'scores': sentiment_scores,
                    'category': category
                })
    else:
        # Nếu câu không có các từ nối đặc biệt, xử lý như bình thường
        part_before_cua = extract_aspect_before_cua(sentence)  # Tách phần trước từ "của"
        aspects = identify_aspects(part_before_cua, aspect_dict) if part_before_cua != sentence else identify_aspects(sentence.lower(), aspect_dict)
        
        if aspects:
            main_aspect = list(aspects.keys())[0]
            main_phrase = aspects[main_aspect]
            
            # Skip if aspect already identified
            if main_aspect not in identified_aspects:
                category = classify_aspect(main_aspect)
                final_sentiment, sentiment_scores = predict_sentiment(main_phrase, main_aspect)
                
                # Mark aspect as identified to avoid duplicates
                identified_aspects.add(main_aspect)
                
                results.append({
                    'overall_sentiment': overall_sentiment,
                    'overall_scores': overall_scores,
                    'phrase': main_phrase,
                    'aspect': main_aspect,
                    'sentiment': final_sentiment,
                    'scores': sentiment_scores,
                    'category': category
                })
    
    # Định dạng đầu ra
    return {
        "result": results,
        "sentence": sentence
    }

# Hàm chính để chạy độc lập
if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:])
    output = infer(sentence)
    print(output)
