import unittest
from unittest.mock import MagicMock, patch
from views import app
import io
import os

class TestAPI(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    @patch('views.model')
    @patch('views.Image.open')
    def test_api_predict_success(self, mock_image_open, mock_model):
        # Mock the model prediction
        mock_model.predict.return_value = 45.0
        
        # Mock image resizing and saving
        mock_img = MagicMock()
        mock_image_open.return_value = mock_img
        
        # Create a dummy image file
        data = {
            'file': (io.BytesIO(b"dummy image data"), 'test.jpg'),
            'model': 'vit'
        }
        
        response = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['angle'], 45.0)
        self.assertEqual(json_data['status'], 'success')

    def test_api_predict_no_file(self):
        response = self.client.post('/api/predict', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No file part', response.get_json()['error'])

if __name__ == '__main__':
    unittest.main()
