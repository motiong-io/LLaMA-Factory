import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from app.services.model_training_service import ModelTrainingService


class TestModelTrainingService(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.service = ModelTrainingService()
        
        # Create a temporary directory for test data
        self.test_dir = "test_data"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('app.repo.minio_client.MinioClient')
    def test_load_dataset_with_progress(self, mock_minio_client):
        """Test that dataset loading shows progress correctly."""
        # Setup mocked MinioClient
        self.service.minio_client = mock_minio_client
        
        # Setup mock for stat_object to return file size
        mock_stat = MagicMock()
        mock_stat.size = 1000  # 1KB file size
        mock_minio_client.stat_object.return_value = mock_stat
        
        # Mock the get_object method to simulate download with progress
        def simulate_download(bucket, obj_name, file_path, callback):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create a simple test file
            with open(file_path, 'wb') as f:
                # Simulate downloading in chunks
                chunk_size = 200
                for i in range(0, 1000, chunk_size):
                    data = b'x' * chunk_size
                    # Call the callback with the chunk
                    callback(data)
                    # Write to the file
                    f.write(data)
            
            return True
        
        mock_minio_client.get_object.side_effect = simulate_download
        
        # Setup paths for test
        minio_path = "test/dataset.json"
        local_path = os.path.join(self.test_dir, "dataset.json")
        
        # Call the method with progress tracking
        with patch('builtins.print') as mock_print, \
             patch('tqdm.tqdm') as mock_tqdm:
            # Setup mock tqdm instance
            mock_tqdm_instance = MagicMock()
            mock_tqdm.return_value = mock_tqdm_instance
            
            # Call the method
            self.service.load_dataset(minio_path, local_path)
            
            # Assert that tqdm was called to create progress bar
            mock_tqdm.assert_called_once()
            
            # Assert update was called multiple times during download
            self.assertTrue(mock_tqdm_instance.update.call_count > 0)
            
            # Assert that progress bar was closed
            mock_tqdm_instance.close.assert_called_once()
            
            # Verify print statements were called for start and complete
            mock_print.assert_any_call(
                f"Downloading dataset from {minio_path}..."
            )
            mock_print.assert_any_call(
                f"Download complete: {local_path}"
            )
    
    @patch('app.repo.minio_client.MinioClient')
    def test_load_dataset_basic(self, mock_minio_client):
        """Test basic dataset loading without checking progress details."""
        # Setup mocked MinioClient
        self.service.minio_client = mock_minio_client
        
        # Setup paths for test
        minio_path = "test/simple_dataset.json"
        local_path = os.path.join(self.test_dir, "simple_dataset.json")
        
        # Call the method
        self.service.load_dataset(minio_path, local_path)
        
        # Verify MinioClient's download_dataset was called with correct args
        mock_minio_client.download_dataset.assert_called_once_with(
            minio_path, local_path, progress_callback=unittest.mock.ANY
        )


if __name__ == "__main__":
    unittest.main() 