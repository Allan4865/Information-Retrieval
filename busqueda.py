from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    keyword = request.args.get('keyword')
    dir = 'data folder' 

    res = []

    for filename in os.listdir(dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                occurrences = re.findall(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE)
                if occurrences:
                    res.append({
                        'filename': filename,
                        'occurrences': len(occurrences),
                        'keywords': ', '.join(occurrences)
                    })

    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
