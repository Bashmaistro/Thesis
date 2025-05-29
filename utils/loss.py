

# Focal loss function
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    import tensorflow as tf
    """
    Focal loss hesaplaması
    :param y_true: Gerçek etiketler
    :param y_pred: Tahmin edilen etiketler
    :param alpha: Dengesizlik ağırlığı (opsiyonel)
    :param gamma: Focal loss'un azalma oranı (opsiyonel)
    :return: Focal loss değeri
    """
    # Tahmin edilen olasılıkları sınıflarla birleştir
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Focal loss hesaplama
    loss = - alpha * (y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)) \
           - (1 - alpha) * ((1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred))

    return tf.reduce_mean(loss)

