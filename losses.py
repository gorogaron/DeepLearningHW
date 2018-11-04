from keras import backend as K

_EPSILON = K.epsilon()

def charbonnier(y_true, y_pred):
    return K.sqrt(K.square(y_true - y_pred) + 0.01**2)

def soft_dice(y_true, y_pred):

    axes = tuple(range(1, len(y_pred.shape))) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean(numerator / (denominator + _EPSILON))