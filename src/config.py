import os

# Paths
DATA_ROOT = "./data/Intracranial_Recordings" 
SCENE_CUTS_CSV = "./data/scenecut_info.csv"
RESULTS_DIR = "./results"

# Data Processing
FS = 1000.0             # Sampling frequency
EEG_MOVIE_START = 10.0  # Offset in seconds
WINDOW_SIZE_SEC = 0.5   # Window size for slicing

# Model Hyperparameters
NUM_CHANNELS = 96
NUM_SAMPLES = 500       # 0.5s * 1000Hz
NUM_CLASSES = 2         # Continuous vs Scene Change
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 20

# Subjects to process
SUBJECT_IDS = list(range(41, 63))