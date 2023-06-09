
# Parameters for generator
BLENDING_LIST = ['simplepaste']


# Parameters for images
MIN_NO_OF_OBJECTS = 2
MAX_NO_OF_OBJECTS = 5
WIDTH = 1920
HEIGHT = 1080
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
# MIN_SCALE = 0.25 # min scale for scale augmentation
# MAX_SCALE = 0.6 # max scale for scale augmentation
MAX_DEGREES = 30 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0.25 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
# MIN_WIDTH = 6 # Minimum width of object to use for data generation
# MIN_HEIGHT = 6 # Minimum height of object to use for data generation