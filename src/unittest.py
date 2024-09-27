import unittest

class TestChurnPredictor(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = {
            'Customer_ID': [1, 2],
            'Contract_Type': ['Month-to-Month', 'Two-Year'],
            'Monthly_Charges': [70, 20],
            'Tenure': [2, 12],
            'Churn_Flag': [1, 0]
        }
        self.predictor = ChurnPredictor(self.data)

    def test_predict_churn(self):
        customer_data = [0, 70, 2]  # Corresponding to Contract_Type, Monthly_Charges, Tenure
        churn_prediction = self.predictor.predict_churn(customer_data)
        self.assertIn(churn_prediction, [0, 1])

    def test_churn_probability(self):
        customer_data = [0, 70, 2]
        probability = self.predictor.churn_probability(customer_data)
        self.assertTrue(0 <= probability <= 1)

if __name__ == '__main__':
    unittest.main()
