import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from classify import classify_text  # Import the classify function from classify.py

# Load test set
test_df = pd.read_csv("D:/Downloads/trustpilot_scraped_dataa.csv")  # columns: description, category

y_true = []
y_pred = []

# Evaluate without calling API (direct function call)
for _, row in test_df.iterrows():
    text = row["description"]
    true_category = row["category"]
    
    try:
        # Use the local classify function instead of API
        predicted_category = classify_text(text)

        y_true.append(true_category)
        y_pred.append(predicted_category)
    except Exception as e:
        print(f"Failed on: {text[:30]}... → {e}")
        y_true.append(true_category)
        y_pred.append("error")

# Accuracy
print("✅ Accuracy:", accuracy_score(y_true, y_pred))

# Detailed report
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred))

