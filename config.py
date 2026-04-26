import os


class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Update this path to your CelebA dataset location
    DATA_DIR = os.path.join(BASE_DIR, 'celeba-data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'img_align_celeba')
    ATTR_FILE = os.path.join(DATA_DIR, 'list_attr_celeba.csv')
    PARTITION_FILE = os.path.join(DATA_DIR, 'list_eval_partition.csv')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

    # Data Settings
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 256
    NUM_WORKERS = 16

    # Target & Sensitive Attribute configurations
    TARGET_ATTR = "Smiling"

    # Multi sensitive attributes for auditing
    SENSITIVE_ATTRS = [
        "Male",
        "Young",
        "Eyeglasses",
        "Wearing_Hat"
    ]

    ENABLE_EPOCH_AUDIT = True

    # Attack Settings (PGD for Training)
    EPSILON = 8 / 255.0
    ROBUST_ALPHA = 2 / 255.0
    ATTACK_STEPS = 10

    # Attack Settings for Evaluation
    EVAL_ATTACK_STEPS = 10

    # Training Hyperparameters
    EPOCHS = 20
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 2e-4
