from flask import Flask, render_template, request
from utils import extract_text, preprocess, plagiarism_detection, writing_style_similarity

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/plagiarism", methods=["GET", "POST"])
def plagiarism():
    
    result = None
    error = None
    if request.method == "POST":
        # Debug: Print what we received
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        print(f"File1: {file1}, filename: {file1.filename if file1 else 'None'}")
        print(f"File2: {file2}, filename: {file2.filename if file2 else 'None'}")

        raw_t1 = extract_text(
            request.form.get("text1"),
            file1
        )
        raw_t2 = extract_text(
            request.form.get("text2"),
            file2
        )

        if not raw_t1:
            error = "Could not extract text from Document 1. Please check the file."
        elif not raw_t2:
            error = "Could not extract text from Document 2. Please check the file."
        else:
            t1 = preprocess(raw_t1)
            t2 = preprocess(raw_t2)

            if t1 and t2:
                result = plagiarism_detection(t1, t2)
            else:
                error = "Text preprocessing resulted in empty content."

    return render_template("plagiarism.html", result=result, error=error)

@app.route("/style", methods=["GET", "POST"])
def style():
    result = None
    if request.method == "POST":
        t1 = preprocess(extract_text(
            request.form.get("text1"),
            request.files.get("file1")
        ))
        t2 = preprocess(extract_text(
            request.form.get("text2"),
            request.files.get("file2")
        ))

        if t1 and t2:
            result = writing_style_similarity(t1, t2)

    return render_template("style.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

