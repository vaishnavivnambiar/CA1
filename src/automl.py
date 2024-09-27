from tpot import TPOTClassifier

class AutoMLChurnPredictor(ChurnPredictor):
    def train_model(self):
        X, y = self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # TPOT AutoML Classifier
        automl = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        automl.fit(X_train, y_train)
        
        # Evaluate
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        
        self.model = automl.fitted_pipeline_
