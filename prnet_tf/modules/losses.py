import tensorflow as tf


def WeightedMSE(uv_weight_mask):
    """MSE loss with mask weighted"""
    def loss_fn(pos, pre):
        loss = (pos - pre) ** 2
        loss = loss * uv_weight_mask
        loss = tf.reduce_mean(loss)
        return loss
    return loss_fn

