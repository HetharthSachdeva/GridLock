import json
import os

def create_notebook():
    print("Building Jupyter Notebook: GridLock_Solution.ipynb...")
    
    # Define notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        })
        
    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source]
        })
        
    # Cell 1: Title
    add_markdown([
        "# Flipkart GridLock Hackathon: Traffic Demand Prediction\n",
        "**Author**: Pair Programming with Antigravity AI\n\n",
        "This notebook contains a complete, production-grade machine learning pipeline to solve the traffic demand forecasting challenge.\n\n",
        "### Solution Architecture Overview:\n",
        "1. **Spatial Features**: Decode `geohash` codes to physical Latitude/Longitude coordinates.\n",
        "2. **Temporal Features**: Convert time string timestamps into continuous minutes-from-midnight and cyclical sine/cosine representations.\n",
        "3. **Historical Demand Matching**: Extract the exact matching spatial-temporal traffic demand from the previous full day (Day 48).\n",
        "4. **Active Trend Deviation**: Compute the localized traffic volume growth/decline ratio between Day 49 and Day 48 morning hours to capture holiday or weather effects.\n",
        "5. **5-Fold Cross-Validation**: Train LightGBM regressors across folds to produce robust, variance-reduced ensembled predictions."
    ])
    
    # Cell 2: Imports
    add_markdown("## 1. Setup & Dependencies")
    add_code([
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pygeohash as gh\n",
        "import lightgbm as lgb\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "import pickle\n",
        "import os\n",
        "import matplotlib.pyplot as plt\n",
        "import warnings\n\n",
        "warnings.filterwarnings('ignore')\n",
        "print('Dependencies loaded successfully!')"
    ])
    
    # Cell 3: Preprocessing Markdown
    add_markdown([
        "## 2. Advanced Feature Engineering & Preprocessing\n",
        "Here we define our advanced preprocessing logic, which extracts structural spatial, temporal, and historical features:"
    ])
    
    # Cell 4: Preprocessing Code (read from src/preprocess.py)
    try:
        with open("src/preprocess.py", "r") as f:
            preprocess_code = f.read()
        # Remove the main block of preprocess.py to make it clean for notebook cells
        main_start = preprocess_code.find("if __name__ == '__main__':")
        if main_start != -1:
            preprocess_code = preprocess_code[:main_start].strip()
    except Exception as e:
        print("Warning: could not read src/preprocess.py directly, using fallback string.", e)
        preprocess_code = "# Preprocessing module not found."
        
    add_code(preprocess_code + "\n\n# Verification run\ntrain_df, test_df, features, target = load_and_preprocess_data()\nprint('\\nFeatures for training:', features)")
    
    # Cell 5: Modeling & Validation Markdown
    add_markdown([
        "## 3. 5-Fold Cross-Validation and Model Training\n",
        "We set up a robust Out-of-Fold (OOF) cross-validation framework to prevent data leakage and train 5 LightGBM Regressors. CatBoost and XGBoost hyperparameters can also be optimized here."
    ])
    
    # Cell 6: Modeling Code (read from src/train.py)
    try:
        with open("src/train.py", "r") as f:
            train_code = f.read()
        # Remove imports and __main__ to avoid redundancy
        main_start = train_code.find("if __name__ == '__main__':")
        if main_start != -1:
            train_code = train_code[:main_start].strip()
            
        # Strip imports that are already done
        lgb_import_idx = train_code.find("from preprocess import")
        if lgb_import_idx != -1:
            # find next line
            next_line_idx = train_code.find("\n", lgb_import_idx)
            train_code = train_code[next_line_idx:].strip()
    except Exception as e:
        print("Warning: could not read src/train.py directly, using fallback string.", e)
        train_code = "# Training module not found."
        
    add_code(train_code + "\n\n# Run the complete training loop!\noof_preds, test_preds = train_model()")
    
    # Cell 7: Feature Importance Markdown
    add_markdown([
        "## 4. Model Interpretation & Feature Importance\n",
        "Let's look at which features are most predictive for traffic demand forecasting:"
    ])
    
    # Cell 8: Feature Importance Code
    add_code([
        "# Load one of the trained fold models to plot feature importance\n",
        "with open('models/lgb_fold_1.pkl', 'rb') as f:\n",
        "    model = pickle.load(f)\n\n",
        "importance = pd.DataFrame({\n",
        "    'Feature': features,\n",
        "    'Importance': model.feature_importance(importance_type='gain')\n",
        "}).sort_values('Importance', ascending=True)\n\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.barh(importance['Feature'], importance['Importance'], color='teal')\n",
        "plt.xlabel('Gain Importance')\n",
        "plt.title('LightGBM Feature Importance')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Cell 9: Submission Markdown
    add_markdown([
        "## 5. Final Submission Verification\n",
        "Let's verify that the output submission file meets all format requirements (correct row count, non-negative demand values, non-null values):"
    ])
    
    # Cell 10: Submission Verification Code
    add_code([
        "sub = pd.read_csv('submission.csv')\n",
        "print('--- Submission Head ---')\n",
        "print(sub.head())\n\n",
        "print('\\n--- Submission Info ---')\n",
        "print(sub.info())\n\n",
        "print('\\nNull count:', sub.isnull().sum().sum())\n",
        "print('Min demand:', sub['demand'].min())\n",
        "print('Max demand:', sub['demand'].max())\n",
        "print('Number of rows matching test set:', len(sub) == 41778)"
    ])
    
    # Write to solution file
    with open("GridLock_Solution.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
        
    print("Notebook compiled and saved to GridLock_Solution.ipynb successfully!")

if __name__ == '__main__':
    create_notebook()
