# CodeLens — Real-Time AI Code Analyzer

CodeLens is a high-performance web-based AI-powered static code-quality and defect analyzer supporting four programming languages:
1. **Python** (`.py`)
2. **C** (`.c`, `.h`)
3. **C++** (`.cpp`, `.cc`, `.hpp`)
4. **Java** (`.java`)

---

## Key Features & Capabilities

### 1. Multi-Syntax Error Detection & Resilient Recovery
- **Multi-Syntax Detection:** Detects independent syntax errors across an entire file rather than aborting at the first encountered error.
- **Delimiter Balance Tracking:** Identifies unclosed or mismatched parentheses `()`, brackets `[]`, and braces `{}` with exact line and column tracking.
- **Python Block-Level Recovery:** Splits modules into top-level functions and classes. If one block has a syntax error, unaffected blocks are parsed and fully analyzed with AST for logic bugs, security flaws, and complexity!

### 2. Deep Language-Specific Analyzers
- **C Analyzer:**
  - Static & loop-based array out-of-bounds access (`CWE-787`, `CWE-193`).
  - Infinite loops (`while(1)`, `for(;;)`, loop variable moving wrong direction).
  - Infinite recursion (unconditional self-call without base-case).
  - Missing return statements in non-void functions (`C-RET01`).
  - Division & modulo by zero (`CWE-369`).
  - Double free (`CWE-415`), use-after-free (`CWE-416`), and memory leaks (`CWE-401`).
  - Uninitialized pointer dereference (`CWE-824`) and NULL dereference (`CWE-476`).
  - Allocation sizing bugs (`sizeof(pointer)` vs `sizeof(*pointer)`).
  - Buffer overflows (`gets()`, unbounded `strcpy()`, `strcat()`, `sprintf()`, `scanf("%s")`).
  - Format string vulnerabilities (`printf(buf)`).
  - GCC compiler syntax & diagnostic integration (`gcc -fsyntax-only -Wall -Wextra`).
- **C++ Analyzer:**
  - `new[]` vs `delete[]` and `new` vs `delete` pairing mismatches (`CPP-DEL-MISMATCH`).
  - Use-after-free and use-after-delete.
  - Heap memory leaks without matching delete or RAII (`CPP-LEAK`).
  - Array and std::vector out-of-bounds access.
  - Invalid std::string indexing (`CPP-STR-BOUNDS`).
  - Dangling stack pointers & references (`return &local_var;`).
  - Unchecked file streams (`std::ifstream`/`ofstream` without `.is_open()`).
  - G++ compiler diagnostic integration (`g++ -fsyntax-only -Wall -Wextra`).
- **Python Analyzer:**
  - Multi-syntax error scanner & indentation error detection.
  - Lexical scoping for undefined variables, symbols, and functions.
  - List index out of range and dictionary `KeyError` detection on literal keys.
  - Type errors (e.g. invalid string and integer concatenation).
  - Division by zero literal and zero-assigned variable tracking.
  - Infinite loops and unconditional infinite recursion.
  - Resource leaks (`open()` without `with` context manager or `.close()`).
  - Security vulnerabilities: `eval()`, `exec()`, SQL injection via string formatting, command injection via `os.system()` and `subprocess(shell=True)`, unsafe pickle deserialization, hardcoded secrets.
  - Ruff linter integration via `python -m ruff`.
- **Java Analyzer:**
  - Reference equality (`==`, `!=`) on String and boxed objects (`PMD-ES01`).
  - Null pointer dereference risks (`JAVA-NULL`).
  - Array out of bounds access (`JAVA-BOUNDS`).
  - Resource leaks on unclosed `Scanner`, `InputStream`, `OutputStream`, etc. without try-with-resources (`JAVA-RES`).
  - Missing return statements on non-void methods (`JAVA-RET01`).
  - Division by zero (`JAVA-ZERO`).
  - Type mismatches and empty catch blocks (`PMD-EMPTY`).
  - Javac compiler diagnostic integration (`javac -Xlint:all`).

### 3. Machine Learning Defect Classifier
- **Dataset:** 602 deduplicated, cleanly labeled samples across all four languages covering 7 defect classes: `clean`, `memory_safety`, `security`, `syntax_error`, `resource_leak`, `control_flow`, and `null_or_bounds`.
- **Pipeline:** Hybrid TF-IDF Word `(1, 2)` + Char `(3, 5)` n-gram feature union with calibrated Logistic Regression classifier.
- **Evaluation:** 5-fold stratified cross validation, 70/15/15 train/val/test split, reporting Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Installation & Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) External Toolchains
# - C / C++: GCC and G++ (e.g. MinGW-w64 on Windows or apt-get install build-essential on Linux)
# - Java: JDK with javac
```

---

## Running the Application

```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## Training the ML Defect Classifier

To retrain the ML model on the curated dataset:
```bash
# Generate the deduplicated dataset
python generate_dataset.py

# Train the model with 5-fold cross validation and test set evaluation
python train_model.py
```

---

## Running Automated Tests

To run the comprehensive test suite verifying detection across all 4 languages:
```bash
python test_comprehensive.py
```
