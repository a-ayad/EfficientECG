from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives(y_true, y_pred) / (possible_positives
                                               + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives(y_true, y_pred) / (predicted_positives
                                                  + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
