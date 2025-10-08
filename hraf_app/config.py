"""
Global configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Global configuration"""

    # API Keys
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Directories
    BASE_DIR = Path(__file__).parent
    CACHE_DIR = BASE_DIR / "cache"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"

    # Ensure directories exist
    for dir_path in [CACHE_DIR, DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        dir_path.mkdir(exist_ok=True)

    # Model defaults
    DEFAULT_BASE_MODEL = "roberta-base"
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 2e-5
    DEFAULT_NUM_EPOCHS = 10

    # Quality scoring defaults
    DEFAULT_K_SIMILAR = 15
    DEFAULT_MIN_QUALITY = 0.60

    # Pinecone
    PINECONE_INDEX_NAME = "hraf-misfortune"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"