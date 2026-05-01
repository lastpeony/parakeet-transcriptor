import logging, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable tqdm progress bars globally (must be set before any tqdm import)
os.environ.setdefault("TQDM_DISABLE", "1")

# --- CPU Threading Configuration (for VAD and audio preprocessing) ---
# Set env vars BEFORE any torch import elsewhere
NUM_THREADS = int(os.getenv("NUM_THREADS", str(os.cpu_count() or 4)))

os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
# Path to a locally-mounted .nemo file. When set, skips network download.
NEMO_MODEL_PATH = os.getenv("NEMO_MODEL_PATH", "")

# Configuration from environment variables
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))          # model’s native sample-rate
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))   # seconds
VAD_THRESHOLD         = float(os.getenv("VAD_THRESHOLD",          "0.35"))
VAD_MIN_SILENCE_MS    = int(os.getenv("VAD_MIN_SILENCE_MS",       "200"))
VAD_SPEECH_PAD_MS     = int(os.getenv("VAD_SPEECH_PAD_MS",        "150"))
VAD_PERIODIC_FLUSH_MS = int(os.getenv("VAD_PERIODIC_FLUSH_MS",    "6000"))
VAD_MIN_CHUNK_MS      = int(os.getenv("VAD_MIN_CHUNK_MS",         "300"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)

# Suppress noisy internal loggers from NeMo and Lhotse.
# Root logger is also set to WARNING so Lhotse's "INFO root: Initializing..."
# lines are silenced — our named loggers have explicit levels and are unaffected.
logging.getLogger().setLevel(logging.WARNING)
for _noisy in ("nemo_logger", "lhotse", "nemo.collections", "nemo.core"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("parakeet_service")
