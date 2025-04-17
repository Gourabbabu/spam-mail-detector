from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        result = "✅ Ham (Not Spam)" if prediction[0] == 1 else "🚨 Spam Detected!"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
