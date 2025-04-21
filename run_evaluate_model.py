from evaluate_model import evaluate_model
from run_load_data import test_features, test_labels

# Total test data count
total_test = 1001  # Total testing data

# Evaluate the model using the test features and labels
evaluate_model(test_features, test_labels, total_test)