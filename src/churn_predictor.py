import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ChurnPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        
    def preprocess(self):
        # Convert categorical 'Contract_Type' to numerical
        self.data['Contract_Type'] = self.data['Contract_Type'].map({'Month-to-Month': 0, 'One-Year': 1, 'Two-Year': 2})
        
        # Define features and target
        X = self.data[['Contract_Type', 'Monthly_Charges', 'Tenure']]
        y = self.data['Churn_Flag']
        
        return X, y

    def train_model(self):
        X, y = self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        
    def predict_churn(self, customer_data):
        customer_data = np.array(customer_data).reshape(1, -1)
        return self.model.predict(customer_data)

    def churn_probability(self, customer_data):
        customer_data = np.array(customer_data).reshape(1, -1)
        return self.model.predict_proba(customer_data)[0][1]
