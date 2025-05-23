import os
from typing import Optional

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str


class LocalDevSettings(EnvSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DeployedSettings(EnvSettings):
    pass


def find_config() -> EnvSettings:
    if os.getenv("ENV"):
        return DeployedSettings()

    return LocalDevSettings()

env = find_config()


if __name__ == "__main__":
    print(env)