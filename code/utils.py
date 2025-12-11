import csv
import logging
import os

# data paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(CURRENT_DIR, "..", "report_src")
DATASETS_DIR = os.path.join(CURRENT_DIR, "..", "datasets")

SHAKESPEARE_DATA_PATH = os.path.join(DATASETS_DIR, "tiny_shakespeare")
PTB_DIR = os.path.join(DATASETS_DIR, "ptb")
WT2_DIR = os.path.join(DATASETS_DIR, "wikitext-2")

LOGS_DIR = os.path.join(CURRENT_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
    ],
)

logger = logging.getLogger(__name__)


csv_path = os.path.join(LOGS_DIR, "training_metrics.csv")
write_header = not os.path.exists(csv_path)

csv_file = open(csv_path, "a", newline="")
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["epoch", "train_loss", "valid_loss"])

# ensure datasets exist(safety!)
if not os.path.exists(SHAKESPEARE_DATA_PATH):
    raise FileNotFoundError(f"Cannot find dataset at {SHAKESPEARE_DATA_PATH}")

if not os.path.exists(PTB_DIR):
    raise FileNotFoundError(f"Cannot find dataset at {PTB_DIR}")

if not os.path.exists(WT2_DIR):
    raise FileNotFoundError(f"Cannot find dataset at {WT2_DIR}")


def load_shakespeare_data(dataset_path):
    train_file = os.path.join(dataset_path, "train.txt")
    valid_file = os.path.join(dataset_path, "valid.txt")
    test_file = os.path.join(dataset_path, "test.txt")

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    return train_text, valid_text, test_text


def load_word_level_data(dataset_dir):
    train_file = os.path.join(dataset_dir, "train.txt")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    return train_text, valid_text, test_text
