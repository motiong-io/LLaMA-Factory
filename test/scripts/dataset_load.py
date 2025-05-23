import os
from app.services.model_training_service import ModelTrainingService
from app.utils.config import config


def dataset_load_test():
    minio_dataset_path = "datasets/LLaMA-Factory/interview-qgen-alpaca-20250516-101021.json"
    local_dataset_path = os.path.join(config.local_storage.path, "datasets/train/interview-qgen-alpaca-20250516-101021.json")

    model_training_service = ModelTrainingService()
    model_training_service.load_dataset(
        minio_dataset_path, local_dataset_path
    )
    return local_dataset_path


if __name__ == "__main__":
    print(dataset_load_test())
