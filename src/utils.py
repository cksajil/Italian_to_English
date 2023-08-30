import tensorflow as tf
from os.path import join
from src import load_config
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

config = load_config("config.yaml")
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def custom_loss(real, pred):
    mask_loss = tf.math.logical_not(tf.math.equal(real, 0))
    loss_metric = loss_object(real, pred)

    mask_loss = tf.cast(mask_loss, dtype=loss_metric.dtype)
    loss_metric *= mask_loss
    return tf.reduce_mean(loss_metric)


def load_pretrained_model():
    """
    Function to load pretrained attention model and return it
    """
    file_path = join(config["model_loc"], config["model_name"])
    model = load_model(file_path, custom_objects={"loss_function": custom_loss})
    return model
