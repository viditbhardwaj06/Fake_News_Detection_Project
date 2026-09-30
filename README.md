# Fake News Detection

This project connects the supplied HTML interface to the four text classifiers implemented in `project.ipynb`. The backend trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models when it starts, then exposes predictions at `/api/predict`.

## Training data

The training CSVs are in the `Datasets` folder beside `app.py`:

```text
Datasets/True.csv
Datasets/Fake.csv
```

Each CSV contains a `text` column. Rows in `True.csv` are labeled real, and rows in `Fake.csv` are labeled fake. The notebook also uses these same files in `Datasets/`.

## Run on Windows

1. Install Python 3.10 or newer and ensure the Python launcher (`py`) is available.
2. Double-click `run.bat`. It creates a local virtual environment, installs dependencies, trains the models, and starts the web server.
3. Open <http://127.0.0.1:5000> in your browser. Keep the terminal open while using the app; press Ctrl+C to stop it.

Alternatively, from PowerShell run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Training can take a few minutes and is repeated on each start. These predictions reflect the dataset and models; they are not independent fact-checks.
