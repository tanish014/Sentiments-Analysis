from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

# Dummy login (you can improve this later)
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username and password:
        return redirect(url_for('analysis'))
    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/guest')
def guest():
    return redirect(url_for('analysis'))

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text')
    # Dummy logic
    sentiment = "Positive" if "good" in text.lower() else "Negative"
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
