from keras.layers import Layer


class PairwiseConnected(Layer):
    def __init__(self, **kwargs):
        super(PairwiseConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.feat_dim = input_shape[-1] // 2
        self.w = self.add_weight(name='weight', shape=(input_shape[-1],),
                                 initializer='uniform', trainable=True)
        super(PairwiseConnected, self).build(input_shape)

    def call(self, x):
        elm_mul = x * self.w
        output = elm_mul[:, 0:self.feat_dim] + elm_mul[:, self.feat_dim:]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.feat_dim
        return tuple(output_shape)
