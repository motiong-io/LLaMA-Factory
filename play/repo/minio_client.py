import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from minio import Minio
from minio.error import S3Error
from play.utils.config import env


class MinioClient(Minio):
    def __init__(self):
        super().__init__(
            endpoint=env.minio_endpoint,
            access_key=env.minio_access_key,
            secret_key=env.minio_secret_key,
            secure=False,
        )

    def upload_dataset(self, 
                       bucket_name: str, 
                       project_name: str,
                       object_name: str, 
                       file_path: str):
        self.fput_object(
            bucket_name=bucket_name,
            object_name=f"datasets/{project_name}/{object_name}",
            file_path=file_path,
        )


    def upload_checkpoint(self,
                     bucket_name: str,
                     model_name: str,
                     training_method: str,
                     training_name_dir: str,
                     local_training_dir: str):
        for root, dirs, files in os.walk(local_training_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_training_dir)
                minio_path = os.path.join(f"saves/{model_name}/{training_method}/{training_name_dir}", relative_path).replace("\\", "/")
                print(f"Uploading {local_path} to {minio_path}")
                self.fput_object(bucket_name, minio_path, local_path)

    def upload_model(self,
                     bucket_name: str,
                     project_name: str,
                     model_name: str,
                     local_model_dir: str):
        for root, dirs, files in os.walk(local_model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_model_dir)
                minio_path = os.path.join(f"models/{project_name}/{model_name}", relative_path).replace("\\", "/")
                print(f"Uploading {local_path} to {minio_path}")
                self.fput_object(bucket_name, minio_path, local_path)



if __name__ == "__main__":
    local_prefix_data = "/home/motiong/hq@sg/LLaMA-Factory/data"
    local_prefix_saves = "/home/motiong/hq@sg/LLaMA-Factory/saves"
    local_prefix_models = "/home/motiong/hq@sg/LLaMA-Factory/model_after_training"
    minio_client = MinioClient()
    print(minio_client.list_buckets())

    # bucket_name = "model-evaluate"
    # project_name = "LLaMA-Factory"
    # object_name = "interview-qgen-alpaca-20250516-101021.json"
    # file_path = f"{local_prefix_data}/{object_name}"

    # minio_client.upload_dataset(
    #     bucket_name=bucket_name,
    #     project_name=project_name ,
    #     object_name=object_name,
    #     file_path=file_path,
    # )

    # bucket_name = "model-evaluate"
    # model_name = "Qwen2-0.5B"
    # training_method = "lora"
    # training_name_dir  = "train_2025-05-18-16-01-01"
    # local_training_dir = f"{local_prefix_saves}/{model_name}/{training_method}/{training_name_dir}"
    
    # minio_client.upload_checkpoint(
    #     bucket_name=bucket_name,
    #     model_name=model_name,
    #     training_method=training_method,
    #     training_name_dir=training_name_dir,
    #     local_training_dir=local_training_dir,
    # )

    bucket_name = "model-evaluate"
    project_name = "LLaMA-Factory"
    model_name = "Qwen2-0.5B-merged-interview-v1"
    local_model_dir = f"{local_prefix_models}/{model_name}"
    minio_client.upload_model(
        bucket_name=bucket_name,
        project_name=project_name,
        model_name=model_name,
        local_model_dir=local_model_dir)