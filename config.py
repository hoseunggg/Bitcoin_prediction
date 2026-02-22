import os
'''

  
├─dataset
│      bitcoin.csv
│      exog_vars.csv
│      preprocess_data.csv
│      raw_data.csv
│
├─figure
│  └─exog
├─report
│      
├─result
│  ├─error
│  │
│  └─rmse
│
├─trash
└─__pycache__
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(BASE_DIR, "dataset")

BITCOIN = os.path.join(DATA_DIR, "bitcoin.csv")
EXOG_VARS = os.path.join(DATA_DIR, "exog_vars.csv")
RAW_DATA = os.path.join(DATA_DIR, "raw_data.csv")
PREPROCESS_DATA = os.path.join(DATA_DIR, "preprocess_data.csv")

RESULT_DIR = os.path.join(BASE_DIR, "result")
RESULT_ERROR_DIR = os.path.join(RESULT_DIR, "error")
RESULT_RMSE_DIR = os.path.join(RESULT_DIR, "rmse")
RESULT_MODEL_DIR = os.path.join(RESULT_DIR, "model")

EXOG_DIR = os.path.join(RESULT_DIR, "exog")
INTERNAL_DIR = os.path.join(RESULT_DIR, "internal")

FIGURE_DIR = os.path.join(BASE_DIR, "figure")
FIGURE_EXOG_DIR = os.path.join(FIGURE_DIR,"exog")


DATA_PREPROCESSING = os.path.join(BASE_DIR, "data_preprocessing.py")

MODEL = os.path.join(BASE_DIR, "model.py")
EVALUATION_PY = os.path.join(BASE_DIR, "evaluation.py")

