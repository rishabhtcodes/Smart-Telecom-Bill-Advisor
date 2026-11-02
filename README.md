# Smart Telecom Bill Advisor — Single-file Project

The complete executable Python implementation is in **main.py**. It connects to MongoDB Atlas with PyMongo, imports the telecom CSV dataset, trains a Decision Tree classifier and a Decision Tree regressor, calculates the six required evaluation metrics, and predicts for one customer.

## Files
- `main.py` — the only Python source file; contains the complete workflow.
- `data/telecom_data.csv` — 7,043 customer records.
- `Project_Report.pdf` — project report.
- `requirements.txt` — Python dependencies.
- `.env.example` — MongoDB URI template.

## Run on Windows / VS Code
1. Extract the ZIP and open the extracted folder in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Create `.env` from `.env.example` and add your MongoDB Atlas URI:
   ```powershell
   Copy-Item .env.example .env
   ```
5. Run the entire project using one command:
   ```powershell
   python main.py
   ```

## MongoDB details
- Database: `telecom_advisor`
- Collection: `telecom_records`

The script clears and reimports only `telecom_advisor.telecom_records` each time it runs. Do not use that collection for unrelated data.

## Models and metrics
- Decision Tree Classifier: Accuracy, Precision, Recall, F1 Score.
- Decision Tree Regressor: R² Score, MAE.

The project creates Low/Medium/High categories using three quantile groups of MonthlyCharges. These are dataset-derived categories, not official telecom tariff bands. The regression target `TotalCharges` is accumulated charges in the dataset, not a guaranteed next invoice.
