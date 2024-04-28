import unittest
import numpy as np
import pandas as pd
from databricks.data_processing import process_genomic_data, process_clinical_data
from models.generative_model import GenerativeModel

class TestMedGenAI(unittest.TestCase):
    def setUp(self):
        self.genomic_data = pd.DataFrame({
            'chromosome': np.random.randint(1, 23, 100),
            'position': np.random.randint(0, 1000000, 100),
            'reference': np.random.choice(['A', 'C', 'G', 'T'], 100),
            'alternate': np.random.choice(['A', 'C', 'G', 'T'], 100),
            'patient_id': np.random.randint(0, 100, 100)
        })
        self.clinical_data = pd.DataFrame({
            'patient_id': np.random.randint(0, 100, 100),
            'diagnosis': np.random.randint(0, 2, 100),
            'age': np.random.randint(0, 100, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'treatment': np.random.choice(['Placebo', 'Drug A', 'Drug B'], 100)
        })
        self.input_shape = (4,)
        self.hidden_units = 64
        self.output_units = 1
        self.model = GenerativeModel(self.input_shape, self.hidden_units, self.output_units)

    def test_data_processing(self):
        X_genomic, y_genomic, patient_ids = process_genomic_data(self.genomic_data)
        X_clinical, y_clinical = process_clinical_data(self.clinical_data)
        self.assertEqual(X_genomic.shape[1], self.input_shape[0])
        self.assertEqual(y_genomic.shape[0], patient_ids.shape[0])
        self.assertEqual(X_clinical.shape[1], self.input_shape[0])
        self.assertEqual(y_clinical.shape[0], patient_ids.shape[0])

    def test_model_training_and_evaluation(self):
        X_genomic, y_genomic, patient_ids = process_genomic_data(self.genomic_data)
        X_clinical, y_clinical = process_clinical_data(self.clinical_data)
        X = np.hstack((X_genomic, X_clinical))
        y = np.hstack((y_genomic, y_clinical))
        self.model.train(X, y, epochs=1, batch_size=32, X_val=X, y_val=y)
        loss, accuracy = self.model.evaluate(X, y)
        self.assertGreater(accuracy, 0.5)

    def test_model_prediction(self):
        X_genomic, y_genomic, patient_ids = process_genomic_data(self.genomic_data)
        X_clinical, y_clinical = process_clinical_data(self.clinical_data)
        X = np.hstack((X_genomic, X_clinical))
        y = np.hstack((y_genomic, y_clinical))
        self.model.train(X, y, epochs=1, batch_size=32, X_val=X, y_val=y)
        X_new = np.random.rand(1, *self.input_shape)
        prediction = self.model.predict(X_new)
        self.assertEqual(prediction.shape, (1, self.output_units))

if __name__ == '__main__':
    unittest.main()
