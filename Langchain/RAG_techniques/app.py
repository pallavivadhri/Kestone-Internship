
from chatbot_rag import (
    run_chatbot2
)


from flask import Flask
from flask import Flask, request, jsonify
app = Flask(__name__)


# @app.route('/')
# def home():
#     return "Hello from local Flask server!"

@app.route('/', methods=['POST'])
def home():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = run_chatbot2(query) 
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True,port=5001)