import os
import random

import numpy as np
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras import layers, models


# =====================================
# DATASET PATHS
# =====================================

DATASET_PATH = r"C:\Users\chand\Downloads\CASIA2 (1)"

AUTHENTIC_PATH = os.path.join(
    DATASET_PATH,
    "Au"
)

MANIPULATED_PATH = os.path.join(
    DATASET_PATH,
    "Tp"
)

MODEL_FOLDER = "models"
MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "image_fraud_model.keras"
)


# =====================================
# SETTINGS
# =====================================

IMAGE_SIZE = 128

# Use a smaller number first for testing.
# Increase later if your system can handle it.
MAX_IMAGES_PER_CLASS = 2000

BATCH_SIZE = 16
EPOCHS = 10

VALIDATION_SPLIT = 0.2

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
)

SEED = 42


# =====================================
# FIND IMAGE FILES
# =====================================

def get_image_files(folder):

    image_files = []

    for root, _, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):

                image_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return image_files


# =====================================
# LOAD IMAGE
# =====================================

def load_image(image_path):

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            # Normalize values from 0-255 to 0-1
            image_array = (
                image_array / 255.0
            )

            return image_array

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        return None


# =====================================
# LOAD DATASET
# =====================================

def prepare_dataset():

    print("\nFinding authentic images...")

    authentic_files = get_image_files(
        AUTHENTIC_PATH
    )

    print(
        "Authentic images found:",
        len(authentic_files)
    )


    print("\nFinding manipulated images...")

    manipulated_files = get_image_files(
        MANIPULATED_PATH
    )

    print(
        "Manipulated images found:",
        len(manipulated_files)
    )


    # Shuffle files
    random.seed(SEED)

    random.shuffle(
        authentic_files
    )

    random.shuffle(
        manipulated_files
    )


    # Limit number of images
    authentic_files = authentic_files[
        :MAX_IMAGES_PER_CLASS
    ]

    manipulated_files = manipulated_files[
        :MAX_IMAGES_PER_CLASS
    ]


    print(
        "\nLoading authentic images..."
    )

    images = []
    labels = []


    for index, image_path in enumerate(
        authentic_files
    ):

        image = load_image(
            image_path
        )


        if image is not None:

            images.append(
                image
            )

            labels.append(
                0
            )


        if (
            (index + 1) % 100 == 0
        ):

            print(
                f"Authentic processed: "
                f"{index + 1}/"
                f"{len(authentic_files)}"
            )


    print(
        "\nLoading manipulated images..."
    )


    for index, image_path in enumerate(
        manipulated_files
    ):

        image = load_image(
            image_path
        )


        if image is not None:

            images.append(
                image
            )

            labels.append(
                1
            )


        if (
            (index + 1) % 100 == 0
        ):

            print(
                f"Manipulated processed: "
                f"{index + 1}/"
                f"{len(manipulated_files)}"
            )


    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )


    # Shuffle final dataset
    indices = np.arange(
        len(images)
    )

    np.random.seed(
        SEED
    )

    np.random.shuffle(
        indices
    )


    images = images[
        indices
    ]

    labels = labels[
        indices
    ]


    print(
        "\nFinal dataset size:",
        len(images)
    )


    return (
        images,
        labels
    )


# =====================================
# BUILD CNN MODEL
# =====================================

def build_model():

    model = models.Sequential([

        layers.Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            )
        ),


        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),


        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),


        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),


        layers.GlobalAveragePooling2D(),


        layers.Dense(
            128,
            activation="relu"
        ),


        layers.Dropout(
            0.4
        ),


        layers.Dense(
            1,
            activation="sigmoid"
        )

    ])


    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model


# =====================================
# TRAIN MODEL
# =====================================

def train():

    print(
        "=" * 50
    )

    print(
        "IMAGE FRAUD DETECTION - MODEL TRAINING"
    )

    print(
        "=" * 50
    )


    # Check dataset folders
    if not os.path.exists(
        AUTHENTIC_PATH
    ):

        print(
            "\nERROR: Authentic folder not found:"
        )

        print(
            AUTHENTIC_PATH
        )

        return


    if not os.path.exists(
        MANIPULATED_PATH
    ):

        print(
            "\nERROR: Manipulated folder not found:"
        )

        print(
            MANIPULATED_PATH
        )

        return


    # Create models folder
    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    # Load data
    images, labels = prepare_dataset()


    if len(images) == 0:

        print(
            "\nERROR: No valid images loaded."
        )

        return


    # Split data
    split_index = int(

        len(images)

        *

        (
            1
            -
            VALIDATION_SPLIT
        )

    )


    x_train = images[
        :split_index
    ]

    y_train = labels[
        :split_index
    ]


    x_validation = images[
        split_index:
    ]

    y_validation = labels[
        split_index:
    ]


    print(
        "\nTraining images:",
        len(x_train)
    )


    print(
        "Validation images:",
        len(x_validation)
    )


    # Build model
    model = build_model()


    print(
        "\nModel Summary:"
    )


    model.summary()


    # Train model
    history = model.fit(

        x_train,

        y_train,

        validation_data=(

            x_validation,

            y_validation

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True

    )


    # Save model
    model.save(
        MODEL_PATH
    )


    print(
        "\n" + "=" * 50
    )

    print(
        "TRAINING COMPLETED SUCCESSFULLY!"
    )

    print(
        "Model saved at:"
    )

    print(
        MODEL_PATH
    )

    print(
        "=" * 50
    )


if __name__ == "__main__":

    train()