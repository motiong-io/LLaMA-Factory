from typing import Any, Dict, Optional

from transformers import TrainerCallback

from app.modules.llamafactory.train.tuner import export_model, run_exp
from app.repo.minio_client import MinioClient


class ModelTrainingService:
    def __init__(self):
        """Initialize the model training service."""
        # For downloading dataset and model saving
        self.minio_client = MinioClient()
        self.args = None
        self.callbacks = []

    def load_dataset(
        self, minio_dataset_path: str, local_dataset_path: str
    ) -> None:
        """Load the dataset.

        Args:
            minio_dataset_path: Path to the dataset in minio
            local_dataset_path: Path to the dataset in local
        """
        print(f"Downloading dataset from {minio_dataset_path}...")
        
        try:
            self.minio_client.download_dataset(
                minio_dataset_path, 
                local_dataset_path,
                )

            print(f"Download complete: {local_dataset_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e

    def set_args(self, args: Dict[str, Any]) -> None:
        """Set training arguments.
        
        Args:
            args: Dictionary of training arguments
        """
        self.args = args
        
    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a training callback.
        
        Args:
            callback: A HuggingFace TrainerCallback instance
        """
        self.callbacks.append(callback)
        
    def train(self) -> None:
        """Run the model training process using LlamaFactory."""
        if not self.args:
            raise ValueError(
                "Training arguments not set. Call set_args() first."
            )
        
        run_exp(args=self.args, callbacks=self.callbacks)
    
    def export(self, export_args: Optional[Dict[str, Any]] = None) -> None:
        """Export the trained model.
        
        Args:
            export_args: Dictionary of export arguments
        """
        args_to_use = export_args if export_args is not None else self.args
        if not args_to_use:
            raise ValueError("Export arguments not provided.")
            
        export_model(args=args_to_use)


