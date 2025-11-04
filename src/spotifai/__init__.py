import os
from dotenv import load_dotenv

load_dotenv()

__project_name__ = "spotifai"
__project_version__ = "0.0.1"
__project_description__ = "Generador de playlist de Spotify con IA"

# SpotifAI config
SPOTIFAI_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".spotifai")
CERT_FILE = f"{SPOTIFAI_CONFIG_DIR}/cert.pem"
KEY_FILE = f"{SPOTIFAI_CONFIG_DIR}/key.pem"

# Spotify API settings
SPOTIFY_CLIENT_ID = "48e756406db640568be5d1b3d6b412e0"

# Spotipy settings
SPOTIPY_REDIRECT_HOST = "127.0.0.1"
SPOTIPY_REDIRECT_PORT = 8888
SPOTIPY_REDIRECT_URI = f"https://{SPOTIPY_REDIRECT_HOST}:{SPOTIPY_REDIRECT_PORT}/callback"
SPOTIPY_CACHE_PATH = os.path.join(SPOTIFAI_CONFIG_DIR, ".cache")

