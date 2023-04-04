from flask import Flask, render_template, jsonify, request
import gzip
import json
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("term")

    # Get the absolute path to city.list.json.gz based on the location of app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    city_list_file = os.path.join(current_dir, "city.list.json.gz")

    with gzip.open(city_list_file, "rt", encoding="utf-8") as f:
        city_data = json.load(f)

    suggestions = [city["name"] for city in city_data if query.lower() in city["name"].lower()][:10]
    return jsonify(suggestions)

if __name__ == "__main__":
    app.run(debug=True)