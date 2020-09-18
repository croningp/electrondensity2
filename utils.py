import tensorflow as tf 

def swish(x):
    """ Swish activation function as described in 
    https://arxiv.org/pdf/1710.05941v1.pdf
    """
    return x * tf.sigmoid(x)



def transorm_ed(density):
    """Transform electron density"""
    density = density + 1e-4
    density = tf.math.log(density)
    density = density / tf.math.log(1e-4)
    
    return density

def transorm_back_ed(density):
    """Back Transform electron density""" 
    density = density * tf.math.log(1e-4)
    density = tf.exp(density) - 1e-4
    return density