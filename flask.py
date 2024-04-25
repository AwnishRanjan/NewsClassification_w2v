from flask import Flask, request, jsonify
from prediction import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Extract the news text from the JSON data
    news_text = data.get('text', '')
    
    # Use the prediction function to get the result
    prediction = predict(news_text)
    
    # Return the prediction as JSON response
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
