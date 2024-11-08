from flask import Flask, request, jsonify
import infer_example  # Import file infer đã xử lý

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get('sentence')
    
    # Sử dụng hàm infer từ infer_example.py để dự đoán sentiment
    prediction_result = infer_example.infer(sentence)
    
    # Tạo response JSON từ kết quả dự đoán
    response = {
        'sentence': sentence,
        'result': prediction_result
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
