from flask import Flask, request, jsonify
import infer_example  # Import file infer đã xử lý

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get('sentence')
    
    # Gọi hàm infer từ infer_example.py và nhận kết quả trả về
    prediction_result = infer_example.infer(sentence)
    
    # Tạo response JSON với cấu trúc mong muốn
    response = {
        "result": prediction_result["result"],  # Chỉ lấy phần "result" từ infer
        "sentence": sentence
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
