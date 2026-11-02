"""Smart Telecom Bill Advisor
Single-file MongoDB + PyMongo + Machine Learning project.
Run: python main.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "telecom_data.csv"


def connect_to_mongodb():
    """Connect to Atlas and return the selected collection and client."""
    load_dotenv(BASE_DIR / ".env")
    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        raise ValueError("MONGO_URI is missing. Add your Atlas URI to the .env file.")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")

    database = client["telecom_advisor"]
    collection = database["telecom_records"]
    return client, database, collection


def import_dataset(collection):
    """Read the CSV file and import its customer records into MongoDB."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
    data = data.astype(object).where(pd.notna(data), None)
    records = data.to_dict("records")

    # This clears only this project's collection before a fresh import.
    collection.delete_many({})
    if records:
        collection.insert_many(records)

    collection.create_index("customerID", unique=True)
    collection.create_index("MonthlyCharges")

    count = collection.count_documents({})
    print("\nMONGODB DATA IMPORT")
    print("Database: telecom_advisor")
    print("Collection: telecom_records")
    print("Records inserted:", count)
    return count


def get_dataframe(collection):
    """Read MongoDB documents into a Pandas DataFrame."""
    documents = list(collection.find({}, {"_id": 0}))
    if not documents:
        raise ValueError("No records found. Import the dataset first.")

    df = pd.DataFrame(documents)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def train_classification_model(df):
    """Train a Decision Tree to classify monthly bills as Low/Medium/High."""
    classification_df = df.dropna(subset=["MonthlyCharges"]).copy()
    classification_df["BillCategory"] = pd.qcut(
        classification_df["MonthlyCharges"],
        q=3,
        labels=["Low", "Medium", "High"],
    )

    features = classification_df.drop(
        columns=[
            "customerID",
            "Churn",
            "MonthlyCharges",
            "TotalCharges",
            "BillCategory",
        ]
    )
    target = classification_df["BillCategory"]

    features = pd.get_dummies(features, drop_first=True).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    return model, list(features.columns), metrics, classification_df


def train_regression_model(df):
    """Train a Decision Tree regressor to estimate accumulated TotalCharges."""
    regression_df = df.dropna(subset=["TotalCharges"]).copy()
    features = regression_df.drop(
        columns=["customerID", "Churn", "TotalCharges", "BillCategory"],
        errors="ignore",
    )
    target = regression_df["TotalCharges"]

    features = pd.get_dummies(features, drop_first=True)
    medians = features.median(numeric_only=True)
    features = features.fillna(medians).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    # Tree regression predicts averages of training targets in its leaves,
    # so it does not extrapolate to negative charges for this positive target.
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
    }
    return model, list(features.columns), medians, metrics, regression_df


def predict_one_customer(
    collection,
    classification_model,
    classification_columns,
    regression_model,
    regression_columns,
    regression_medians,
):
    """Generate both predictions for one customer stored in MongoDB."""
    customer = collection.find_one(
        {"tenure": {"$gte": 1}},
        {"_id": 0},
        sort=[("tenure", -1)],
    )
    if customer is None:
        print("No suitable customer record was found for the demonstration.")
        return

    customer_df = pd.DataFrame([customer])

    classification_input = customer_df.drop(
        columns=["customerID", "Churn", "MonthlyCharges", "TotalCharges"],
        errors="ignore",
    )
    classification_input = pd.get_dummies(
        classification_input, drop_first=False
    ).fillna(0)
    classification_input = classification_input.reindex(
        columns=classification_columns, fill_value=0
    )
    bill_category = classification_model.predict(classification_input)[0]

    regression_input = customer_df.drop(
        columns=["customerID", "Churn", "TotalCharges", "BillCategory"],
        errors="ignore",
    )
    regression_input = pd.get_dummies(regression_input, drop_first=False)
    regression_input = regression_input.reindex(
        columns=regression_columns
    )
    regression_input = regression_input.fillna(regression_medians).fillna(0)
    estimated_charges = regression_model.predict(regression_input)[0]

    print("\nSAMPLE CUSTOMER PREDICTION")
    print("Customer ID:", customer.get("customerID", "Not available"))
    print("Predicted monthly bill category:", bill_category)
    print("Estimated total charges:", round(float(estimated_charges), 2))
    print("Note: TotalCharges is accumulated charges in the dataset, not a guaranteed next invoice.")


def main():
    """Run the complete project from one Python file."""
    client = None
    try:
        client, database, collection = connect_to_mongodb()
        print("Connected to MongoDB Atlas successfully.")

        record_count = import_dataset(collection)
        df = get_dataframe(collection)

        print("\nTRAINING MACHINE LEARNING MODELS")
        classification_model, classification_columns, class_metrics, class_df = (
            train_classification_model(df)
        )
        regression_model, regression_columns, regression_medians, reg_metrics, reg_df = (
            train_regression_model(class_df)
        )

        print("Records used:", len(class_df))
        print("\nBILL CATEGORY CLASSIFICATION - DECISION TREE")
        for metric, value in class_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nTOTAL CHARGES REGRESSION - DECISION TREE REGRESSOR")
        for metric, value in reg_metrics.items():
            print(f"{metric}: {value:.4f}")

        predict_one_customer(
            collection,
            classification_model,
            classification_columns,
            regression_model,
            regression_columns,
            regression_medians,
        )

        print("\nPROJECT COMPLETED SUCCESSFULLY.")
        print("Database:", database.name)
        print("Imported records:", record_count)

    except Exception as error:
        print("\nPROJECT ERROR:", error)
        print("Check your .env connection string, Atlas Network Access and dataset path.")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
