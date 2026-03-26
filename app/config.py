from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # WeChat
    wechat_token: str = ""
    wechat_app_id: str = ""
    wechat_app_secret: str = ""
    wechat_encoding_aes_key: str = ""

    # AI
    ai_api_key: str = ""
    ai_base_url: str = "https://api.deepseek.com"
    ai_model: str = "deepseek-chat"
    ai_vision_model: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # Customer service API (certified accounts only)
    enable_customer_service_api: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
