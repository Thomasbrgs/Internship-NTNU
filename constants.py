# Load times
BEFORE_MARKER_TIME = 3
MARKER_TIME = 3
AFTER_MARKER_TIME = 3
TOTAL_TIME = BEFORE_MARKER_TIME + MARKER_TIME + AFTER_MARKER_TIME
SAMPLE_WINDOW = BEFORE_MARKER_TIME+MARKER_TIME

# Data stream
# STREAM_NAME = "apple_game"
STREAM_NAME = "Explore_8547_ExG"

# Game settings
FPS = 30
END_VALUE = 20 

# Sizes
scale = 1.9
SCREEN_WIDTH = 1000*scale
SCREEN_HEIGHT = 750*scale
PLAYER_WIDTH = 200*scale
PLAYER_HEIGHT = 200*scale
DRONE_SIZE = 200*scale
LOAD_BAR_HEIGHT = 50*scale

# Colors
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_COLOR = (0, 0, 0)
DRONE_COLOR = (255, 0, 0)
LOAD_BAR_COLOR = (0, 255, 0)  # Green color for the load bar
MARKER_BAR_COLOR = (255, 128, 0)  # Orange color for the marker line

# Image paths
LEFT_HAND_OPEN_PATH = "images/left_hand_open.png" 
LEFT_HAND_CLOSED_PATH = "images/left_hand_closed.png"  
RIGHT_HAND_OPEN_PATH = "images/right_hand_open.png"  
RIGHT_HAND_CLOSED_PATH = "images/right_hand_closed.png"  
DRONE_IMAGE_PATH = "/home/thomas/Stage_2A/final /drone.png"  
TREE_IMAGE_PATH = "/home/thomas/Stage_2A/final /ville.png"

# Channel names for different EEG devices
CH_NAMES_4 = ['C3', 'C4', 'Cz', 'Pz']
CH_NAMES_8 = ['C5', 'C6', 'C3', 'C1', 'FC3', 'C4', 'C2', 'FC4']
CH_NAMES_32 = [
    "F3", # 1
    "F1", # 2
    "Fz", # 3
    "FC1", # 4
    "FCz", # 5
    "Cz", # 6
    "P5", # 7
    "CP5", # 8
    "CP3", # 9
    "CP1", # 10
    "FC3", # 11
    "FC5", # 12
    "C5", # 13
    "C3", # 14
    "C1", # 15
    "P1", # 16
    "C4", # 17
    "C6", # 18
    "CP6", # 19
    "CP4", # 20
    "CP2", # 21
    "CPz", # 22
    "AFz", # 23
    "F4", # 24
    "F2", # 25
    "C2", # 26
    "FC6", # 27
    "FC4", # 28
    "FC2", # 29
    "Pz", # 30
    "P2", # 31
    "P6" # 32
] 
CH_NAMES_64 =[
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]

# Preprocessing constants
F_LOW = 1
F_HIGH = 50
F_NOTCH = 50
# F_BANDS = [(4,8),(8,12),(12,20),(20,30),(30,45)]
F_BANDS = [(7,11),(9,13)]


# Thresolds
CONFIDENCE_THRESHOLD = 0.7
