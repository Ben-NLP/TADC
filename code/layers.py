from keras.engine.topology import Layer
from keras import activations, initializers, optimizers
from keras.layers import Input, Dense, Lambda, Embedding, Dropout, Conv1D
from keras.layers.merge import add
from keras.models import Model
import keras.backend as K
import numpy as np
import random

class ContextEmb(Layer):
    def __init__(self, context=1, **kwargs):
        self.context = context
        super(ContextEmb, self).__init__(**kwargs)
    
    def call(self, x):
        x1 = K.temporal_padding(x, padding=(1, 0))
        x1 = x1[:,0:-1,:]
        x2 = K.temporal_padding(x, padding=(0, 1))
        x2 = x2[:,1:,:]
        x = K.concatenate([x, x1, x2])
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Decoder(Layer):
    def __init__(self, weights, **kwargs):
        self.topic_emb, self.bow_emb = weights
        self.hidden_size = self.topic_emb.shape[1]
        super(Decoder, self).__init__(**kwargs)

    def my_add_weight(self, name, weights, trainable=True):
        weight = K.variable(weights, name=name)
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight

    def build(self, input_shape):
        self.t = self.my_add_weight(name='{}_T'.format(self.name),
                                    weights=self.topic_emb,
                                    trainable=True)
        self.v = self.my_add_weight(name='{}_W'.format(self.name),
                                    weights=self.bow_emb,
                                    trainable=True)
        self.built = True

    def call(self, x):
        tv = K.dot(self.t, K.transpose(self.v))
        tv = K.softmax(tv, axis=-1)
        theta = K.softmax(x, axis=-1)
        x = K.dot(theta, tv)

        return [x, theta, self.t]

    def compute_output_shape(self, input_shape):
        bow_maxlen = self.bow_emb.shape[0]
        topic_num = self.topic_emb.shape[0]
        return [(input_shape[0], bow_maxlen), (input_shape[0], topic_num), (topic_num, self.hidden_size)]

class LowerBound(Layer):
    def __init__(self, **kwargs):
        super(LowerBound, self).__init__(**kwargs)

    def call(self, input_tensor):
        z_mean, z_log_var, bow_input, bow_output, bow_mask = input_tensor

        kl = -0.5 * K.sum(1 - K.square(z_mean) +  z_log_var - K.exp(z_log_var), axis=-1)
        kl *= bow_mask

        nnl = -K.sum(bow_input * K.log(bow_output + K.epsilon()), axis=-1)
        nnl *= bow_mask

        return [kl, nnl]

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return [(batch_size,1), (batch_size,1)]

class Regularization(Layer):
    def __init__(self, **kwargs):
        super(Regularization, self).__init__(**kwargs)

    def call(self, x):
        size = K.int_shape(x)[0]

        l2 = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)) + K.epsilon()
        x = x / l2
        x = K.dot(x, K.transpose(x))
        x = x - K.eye(size)

        reg = K.sum(K.square(x))

        return reg

    def compute_output_shape(self, input_shape):
        return ()

class BowMasking(Layer):
    def __init__(self, **kwargs):
        super(BowMasking, self).__init__(**kwargs)

    def call(self, x):
        mask = K.cast(K.any(x, axis=-1), K.dtype(x))
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

class SeqMasking(Layer):
    def __init__(self, **kwargs):
        super(SeqMasking, self).__init__(**kwargs)

    def call(self, x):
        mask = K.greater(x, 0)
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor):
        x, e = input_tensor
        z = K.sum(x*e, axis=1)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

class Attention(Layer):
    def __init__(self, kernel_init='glorot_uniform', bias_init='zeros', **kwargs):
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        dim1 = input_shape[0][-1]*2 + input_shape[1][-1]
        dim2 = input_shape[0][-1]
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(dim1, dim2),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.u = self.add_weight(name='{}_v'.format(self.name),
                                 shape=(dim2, 1),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.b = self.add_weight(name='{}_b'.format(self.name),
                                 shape=(1, dim2),
                                 initializer=self.bias_init,
                                 trainable=True)
        self.built = True

    def call(self, input_tensor):
        x, h, mask = input_tensor

        mask = K.cast(mask, K.dtype(x))

        temp = K.equal(mask, 0)
        temp = K.cast(temp, K.dtype(x))
        temp = K.expand_dims(temp, axis=-1)
        temp *= -1000000

        y = x + temp
        p = K.max(y, axis=1)

        rep = K.int_shape(x)[1]

        p1 = K.expand_dims(p, axis=1)
        p1 = K.repeat_elements(p1, rep=rep, axis=1)

        h = K.expand_dims(h, axis=1)
        h = K.repeat_elements(h, rep=rep, axis=1)

        tmp = K.concatenate([p1, x, h], axis=-1)
        b = K.expand_dims(self.b, axis=0)

        d = K.dot(tmp, self.W)
        d = K.tanh(d+b)
        d = K.dot(d, self.u)

        e = K.exp(d)
        mask = K.expand_dims(mask, axis=-1)
        mask = K.cast(mask, K.dtype(e))
        e *= mask
        e /= K.cast(K.sum(e, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return [p, e]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][-1]), (input_shape[0][0], input_shape[0][1], 1)]

class Attention2(Layer):
    def __init__(self, kernel_init='glorot_uniform', bias_init='zeros', **kwargs):
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        super(Attention2, self).__init__(**kwargs)

    def build(self, input_shape):
        dim1 = input_shape[0][-1]*2 + input_shape[3][-1]
        dim2 = input_shape[0][-1]
        self.P = self.add_weight(name='{}_V'.format(self.name),
                                 shape=(dim2, dim2),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(dim1, dim2),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.u = self.add_weight(name='{}_v'.format(self.name),
                                 shape=(dim2, 1),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.b = self.add_weight(name='{}_b'.format(self.name),
                                 shape=(1, dim2),
                                 initializer=self.bias_init,
                                 trainable=True)
        self.built = True

    def call(self, input_tensor):
        x, p, z, h, mask = input_tensor

        p = K.transpose(K.dot(self.P, K.transpose(p)))
        p = p + z

        rep = K.int_shape(x)[1]

        p1 = K.expand_dims(p, axis=1)
        p1 = K.repeat_elements(p1, rep=rep, axis=1)

        h = K.expand_dims(h, axis=1)
        h = K.repeat_elements(h, rep=rep, axis=1)

        tmp = K.concatenate([p1, x, h], axis=-1)
        b = K.expand_dims(self.b, axis=0)

        d = K.dot(tmp, self.W)
        d = K.tanh(d+b)
        d = K.dot(d, self.u)

        e = K.exp(d)
        mask = K.expand_dims(mask, axis=-1)
        mask = K.cast(mask, K.dtype(e))
        e *= mask
        e /= K.cast(K.sum(e, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        return [p, e]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][-1]), (input_shape[0][0], input_shape[0][1], 1)]

class DyCNN(Layer):
    def __init__(self, filters, kernel_size, padding, use_bias, activation, kernel_init='glorot_uniform', bias_init='zeros', **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        super(DyCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.din = input_shape[0][-1]

        self.U = self.add_weight(name='{}_P'.format(self.name),
                                 shape=(self.din, self.din),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.V = self.add_weight(name='{}_Q'.format(self.name),
                                 shape=(self.din, self.kernel_size*self.filters),
                                 initializer=self.kernel_init,
                                 trainable=True)
        self.B = self.add_weight(name='{}_B'.format(self.name),
                                 shape=(self.din, self.kernel_size*self.filters),
                                 initializer=self.kernel_init,
                                 trainable=True)
        if self.use_bias:
             self.b = self.add_weight(name='{}_b'.format(self.name),
                                      shape=(self.filters,),
                                      initializer=self.bias_init,
                                      trainable=True)
        self.built = True

    def call(self, input_tensor):
        x, z = input_tensor

        U = K.expand_dims(self.U, axis=0)
        z = K.sigmoid(z)
        z = K.expand_dims(z, axis=1)

        f = U * z
        f = K.dot(f, self.V)

        B = K.expand_dims(self.B, axis=0)
        f = f + B

        f = K.reshape(f, (-1, self.din, self.kernel_size, self.filters))
        f = K.permute_dimensions(f, (0, 2, 1, 3))

        def single_conv(tupl):
            inp, kernel = tupl
            outputs = K.conv1d(inp, kernel, padding=self.padding)
            if self.use_bias:
                outputs = K.bias_add(outputs, self.b)
            if self.activation is not None:
                return self.activation(outputs)
            return outputs

        res = K.squeeze(K.map_fn(single_conv, (K.expand_dims(x, 1), f), dtype=K.floatx()), axis=1)

        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.filters)
    
def sampling(args):
    mean, log_var = args
    topic_num = K.int_shape(mean)[-1]
    epsilon = K.random_normal(shape=(topic_num,), mean=0., stddev=1.)
    return mean + K.exp(log_var / 2) * epsilon

def kl_loss(y_true, y_pred):
    return y_pred

def nnl_loss(y_true, y_pred):
    return y_pred

def reg_loss(y_true, y_pred):
    return y_pred

def create_model(bow_maxlen, hidden_size, topic_num, shortcut, seq_maxlen, dropout, lr, topic_emb, bow_emb, seq_emb):
    np.random.seed(1337)
    random.seed(1337)
    ### cnn layers ###
    gen_emb = Embedding(seq_emb.shape[0], seq_emb.shape[1], weights=[seq_emb], trainable=False, name='gen_emb')
    conv1 = Conv1D(filters=300, kernel_size=3, padding='same', use_bias=True, activation='relu', name='conv1')
    conv2 = DyCNN(filters=300, kernel_size=5, padding='same', use_bias=True, activation='relu', name='conv2')
    conv3 = Conv1D(filters=300, kernel_size=5, padding='same', use_bias=True, activation='relu', name='conv3')
    conv4 = DyCNN(filters=300, kernel_size=5, padding='same', use_bias=True, activation='relu', name='conv4')
    linear = Dense(3, activation='softmax', name='linear')
    
    att1 = Attention(name='att1')
    att2 = Attention2(name='att2')

    ### ntm layers ###
    e1 = Dense(hidden_size, activation='relu', name='e1')
    e2 = Dense(hidden_size, activation='relu', name='e2')
    es = Dense(hidden_size, name='es')
    e3 = Dense(topic_num, name='e3')
    e4 = Dense(topic_num, name='e4')

    g1 = Dense(topic_num, activation='relu', name='g1')
    g2 = Dense(topic_num, activation='relu', name='g2')
    g3 = Dense(topic_num, activation='relu', name='g3')
    g4 = Dense(topic_num, activation='relu', name='g4')

    ### ntm graph ###
    bow_input = Input(shape=(bow_maxlen,), name='bow_input')
    bow_mask = BowMasking()(bow_input)

    h = e1(bow_input)
    h = e2(h)
    if shortcut:
        h = add([h, es(bow_input)])

    z_mean = e3(h)
    z_log_var = e4(h)
    hidden = Lambda(sampling)([z_mean, z_log_var])

    tmp = g1(hidden)
    tmp = g2(tmp)
    tmp = g3(tmp)
    tmp = g4(tmp)
    if shortcut:
        tmp = add([tmp, hidden])

    d1 = Decoder(weights=[topic_emb, bow_emb], name='d1')
    bow_output, theta, tv = d1(tmp)

    kl, nnl = LowerBound()([z_mean, z_log_var, bow_input, bow_output, bow_mask])
    reg = Regularization()(tv)

    ### cnn graph ###
    seq_input = Input(shape=(seq_maxlen,), name='seq_input')
    seq_mask = SeqMasking()(seq_input)

    x = gen_emb(seq_input)
    x = Dropout(dropout)(x)

    x = conv1(x)
    p, e = att1([x, theta, seq_mask])
    z = WeightedSum()([x, e])
    x = Dropout(dropout)(x)

    x = conv2([x, z])
    x = Dropout(dropout)(x)

    x = conv3(x)
    p, e = att2([x, p, z, theta, seq_mask])
    z = WeightedSum()([x, e])
    x = Dropout(dropout)(x)

    x = conv4([x, z])

    pred = linear(x)

    ### combine ###
    model = Model(inputs=[bow_input, seq_input], outputs=[kl, nnl, reg, pred])
    adam = optimizers.Adam(lr=lr, clipnorm=1.)
    model.compile(optimizer=adam, loss=[kl_loss, nnl_loss, reg_loss, 'categorical_crossentropy'], loss_weights=[1., 1., 1., 1.])

    return model