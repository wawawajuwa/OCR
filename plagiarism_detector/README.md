# Text Analysis Application

A Flask-based web application for plagiarism detection and writing style comparison.

## Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer
- Tesseract OCR (for image/scanned PDF text extraction)
- Poppler (for PDF to image conversion)

---

## Windows 10 Setup

### 1. Install Python

Download and install Python 3.8+ from https://www.python.org/downloads/

During installation, check **"Add Python to PATH"**.

### 2. Install uv

Open PowerShell and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:
```cmd
pip install uv
```

### 3. Install Tesseract OCR

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (use default path: `C:\Program Files\Tesseract-OCR`)
3. During installation, select additional languages if needed (e.g., Indonesian)

### 4. Install Poppler

1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the ZIP file to `C:\poppler`
3. Add Poppler to PATH:
   - Open **System Properties** > **Environment Variables**
   - Under **System variables**, find **Path** and click **Edit**
   - Add `C:\poppler\Library\bin`
   - Click **OK** to save

### 5. Setup Project

Open Command Prompt or PowerShell:

```cmd
cd project_ai

# Create virtual environment
uv venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
uv pip install flask numpy pillow pytesseract pypdf pdf2image scikit-learn
```

### 6. Run the Application

```cmd
python app.py
```

Open browser at http://127.0.0.1:5000

---

## macOS Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Tesseract OCR and Poppler

```bash
brew install tesseract poppler
```

### 3. Setup Project

```bash
cd project_ai

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install flask numpy pillow pytesseract pypdf pdf2image scikit-learn
```

### 4. Run the Application

```bash
python app.py
```

---

## Ubuntu/Debian Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Tesseract OCR and Poppler

```bash
sudo apt install tesseract-ocr poppler-utils
```

### 3. Setup Project

```bash
cd project_ai

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install flask numpy pillow pytesseract pypdf pdf2image scikit-learn
```

### 4. Run the Application

```bash
python app.py
```

---

## Features

### Plagiarism Detection (`/plagiarism`)
Compare two documents with detailed analysis:
- Semantic similarity (TF-IDF cosine)
- Word overlap (Jaccard similarity)
- Phrase matching (N-gram analysis)
- Matching phrases detection
- Risk level interpretation

### Writing Style Analysis (`/style`)
Compare writing styles with breakdown:
- Sentence structure
- Word complexity
- Vocabulary richness
- Punctuation style
- Expression style

### Supported Input Formats
- Direct text input
- PDF files (text-based and scanned/image-based with OCR)
- Image files (PNG, JPG, JPEG) with OCR
