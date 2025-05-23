import os
from configparser import ConfigParser
from functools import cached_property
from pathlib import Path
from typing import TypeVar, cast

T = TypeVar("T", str, bool, int, float)


class ConfigValue:
    def __init__(self, section: str):
        self._section = section
        self._config = ConfigParser()

        # Load the INI file if it exists
        config_path = "config.ini"
        if Path(config_path).exists():
            self._config.read(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def get_value(self, key: str, value_type: type[T]) -> T:
        """
        Get a configuration value with the following precedence:
        1. Environment variable (SECTION_KEY)
        2. INI file value
        """
        # First check environment variable
        env_key = f"{self._section.upper()}_{key.upper()}"
        env_value = os.environ.get(env_key)

        if env_value is not None:
            return self._cast_value(env_value, value_type)

        # Fallback to INI file
        value = self._config.get(self._section, key)
        os.environ[env_key] = value
        return self._cast_value(value, value_type)

    def _cast_value(self, value: str, value_type: type[T]) -> T:
        """Cast the string value to the specified type"""
        if value_type == bool:
            result = value.lower() in ("true", "1", "yes", "on")
            return cast(T, result)
        if value_type == int:
            return cast(T, int(value))
        if value_type == float:
            return cast(T, float(value))
        if value_type == str:
            return cast(T, value)
        raise ValueError(f"Unsupported type: {value_type}")


class _ServiceConfig(ConfigValue):
    def __init__(self):
        super().__init__("service")

    @cached_property
    def name(self) -> str:
        return self.get_value("name", str)

    @cached_property
    def version(self) -> str:
        return self.get_value("version", str)

    def __str__(self) -> str:
        return f"Name: {self.name} Version: {self.version}"


class _LocalStorageConfig(ConfigValue):
    def __init__(self):
        super().__init__("local_storage")

    @cached_property
    def path(self) -> str:
        return self.get_value("path", str)

    def __str__(self) -> str:
        return f"Path: {self.path}"


class _MinioConfig(ConfigValue):
    def __init__(self):
        super().__init__("minio")

    @cached_property
    def endpoint(self) -> str:
        return self.get_value("endpoint", str)

    @cached_property
    def username(self) -> str:
        return self.get_value("username", str)

    @cached_property
    def password(self) -> str:
        return self.get_value("password", str)
    
    @cached_property
    def bucket_name(self) -> str:
        return self.get_value("bucket_name", str)

    def __str__(self) -> str:
        return (f"Endpoint: {self.endpoint} Username: {self.username} "
                f"Password: {self.password} Bucket Name: {self.bucket_name}")


class _LLMR1Config(ConfigValue):
    def __init__(self):
        super().__init__("llm_r1")

    @cached_property
    def endpoint(self) -> str:
        return self.get_value("endpoint", str)

    @cached_property
    def model(self) -> str:
        return self.get_value("model", str)

    @cached_property
    def api_key(self) -> str:
        return self.get_value("api_key", str)

    def __str__(self) -> str:
        return (f"Endpoint: {self.endpoint} Model: {self.model} "
                f"API Key: {self.api_key}")


class _LLMLlamaConfig(ConfigValue):
    def __init__(self):
        super().__init__("llm_llama")

    @cached_property
    def endpoint(self) -> str:
        return self.get_value("endpoint", str)

    @cached_property
    def model(self) -> str:
        return self.get_value("model", str)

    @cached_property
    def api_key(self) -> str:
        return self.get_value("api_key", str)

    def __str__(self) -> str:
        return (f"Endpoint: {self.endpoint} Model: {self.model} "
                f"API Key: {self.api_key}")


class _Config:
    service = _ServiceConfig()
    local_storage = _LocalStorageConfig()
    minio = _MinioConfig()
    llm_r1 = _LLMR1Config()
    llm_llama = _LLMLlamaConfig()

    def __str__(self) -> str:
        return (f"Loaded config: Service: {self.service} "
                f"Local Storage: {self.local_storage} "
                f"Minio: {self.minio} LLM R1: {self.llm_r1} "
                f"LLM Llama: {self.llm_llama}")


config = _Config()


def load_config() -> None:
    # Go through all the config values once so the cached properties are 
    # populated
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        print(config, file=devnull)


__all__ = ["config", "load_config"]