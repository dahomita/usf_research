from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load research data
df = pd.read_excel("files/articles.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip().str.lower()
df["text_blob"] = df.apply(lambda row: f"Title: {row['title']}\nYear: {row['year published']}\nType: {row['type']}\nAbstract: {row['abstract']}\nKeywords: {row['keywords']}", axis=1)
data_blob = "\n\n".join(df["text_blob"].tolist())

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat_interface.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    prompt = f"Dataset:\n{data_blob[:12000]}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing academic research metadata on AI ethics."},
            {"role": "user", "content": prompt}
        ]
    )

    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)
