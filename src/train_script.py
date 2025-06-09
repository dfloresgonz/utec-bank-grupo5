import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import glob


def model_fn(model_dir):
  """Load model for inference - REQUIRED for SageMaker endpoints"""
  model = joblib.load(os.path.join(model_dir, "model.joblib"))
  return model


if __name__ == "__main__":
  try:
    print("=== Starting training script ===")

    data_dir = os.environ.get('SM_CHANNEL_DATA', '/opt/ml/input/data/data')
    print(f"Looking for data in: {data_dir}")

    csv_file = glob.glob(f"{data_dir}/*.csv")[0]
    print(f"Found CSV file: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    df = df.dropna()
    print(f"Data shape after dropna: {df.shape}")

    X = df.drop("ATTRITION", axis=1)
    y = df["ATTRITION"]

    # Convert categorical columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
      X[col] = pd.Categorical(X[col]).codes

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print(f"METRIC train_accuracy {train_accuracy}")
    print(f"METRIC test_accuracy {test_accuracy}")

    # Guardar modelo
    print("Saving model...")
    os.makedirs("/opt/ml/model", exist_ok=True)
    joblib.dump(model, "/opt/ml/model/model.joblib")

    print("✅ Training completed successfully!")

  except Exception as e:
    print(f"❌ Training failed with error: {str(e)}")
    import traceback
    traceback.print_exc()
    raise e
