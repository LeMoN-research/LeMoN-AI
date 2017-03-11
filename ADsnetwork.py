#get_ipython().magic('env THEANO_FLAGS=device=gpu0,floatX=float32')#
import theano
import theano.tensor as T
import lasagne

MUSIC_SHAPE = (None, 20, 200)
START_POSITION_SHAPE = (None, 38, 3)
SHIFT_SHAPE = (None, 19, 38*3)


class LeMoN_AI(object):

    def __init__(self, trainable=True):
        input_music_var = T.tensor3("Music input")
        input_shift_var = T.tensor3("Shift input")

        input_position_var = T.tensor3("Start position")
        delta_mov_var = T.matrix("Delta moving")
        
        music_input = lasagne.layers.InputLayer(shape=MUSIC_SHAPE, input_var=input_music_var)
        music_dims = lasagne.layers.DimshuffleLayer(music_input, (0, 2, 1))
        print("Music conv: ", music_dims.output_shape)
        music_conv = lasagne.layers.Conv1DLayer(music_dims, 256, 10, name="Music")
        music_conv = lasagne.layers.batch_norm(music_conv)
        # music_pool = lasagne.layers.GlobalPoolLayer(music_conv, T.max)
        print("Music conv: ", music_conv.output_shape)

        shift_input = lasagne.layers.InputLayer(shape=SHIFT_SHAPE, input_var=input_shift_var)
        shift_dims = lasagne.layers.DimshuffleLayer(shift_input, (0, 2, 1))
        print("Shift conv: ", shift_dims.output_shape)
        shift_conv = lasagne.layers.Conv1DLayer(shift_dims, 256, 10, name="shift_conv")
        shift_conv = lasagne.layers.batch_norm(shift_conv)
        # shift_pool = lasagne.layers.GlobalPoolLayer(music_conv, T.max)
        print("Shift conv: ", shift_conv.output_shape)

        position_input = lasagne.layers.InputLayer(shape=START_POSITION_SHAPE, input_var=input_position_var)
        position_dims = lasagne.layers.DimshuffleLayer(position_input, (0, 2, 1))
        position_conv = lasagne.layers.Conv1DLayer(position_dims, 128, 2, name="Conv0")
        position_conv = lasagne.layers.batch_norm(position_conv)
        position_pool = lasagne.layers.Pool1DLayer(position_conv, 2, name="Pool0")
        position_conv = lasagne.layers.Conv1DLayer(position_pool, 512, 2, name="Conv1")
        position_conv = lasagne.layers.batch_norm(position_conv)
        position_pool = lasagne.layers.GlobalPoolLayer(position_conv, pool_function=T.max, name="Gpool")
        print("Position pool: ", position_pool.output_shape)

        lstm_music = lasagne.layers.LSTMLayer(music_conv, 512, only_return_final=True, name="Music LSTM")
        lstm_music = lasagne.layers.batch_norm(lstm_music)
        # lstm_music = lasagne.layers.ReshapeLayer(lstm_music, (-1, 1, 512))

        lstm_shift = lasagne.layers.LSTMLayer(shift_conv, 512, hid_init=position_pool, only_return_final=True, name="Shift LSTM")
        lstm_shift = lasagne.layers.batch_norm(lstm_shift)
        # lstm_shift = lasagne.layers.ReshapeLayer(lstm_shift, (-1, 1, 512))

        conc = lasagne.layers.ConcatLayer([lstm_music, lstm_shift], axis=1)
        print("Conc: ", conc.output_shape)
        dense_0 = lasagne.layers.DenseLayer(conc, 256, nonlinearity=lasagne.nonlinearities.sigmoid, name="Dense1")
        output = lasagne.layers.DenseLayer(dense_0, 38*3, nonlinearity=lasagne.nonlinearities.linear, name="Conv2")
        print("Output: ", output.output_shape)

        reshape_output = lasagne.layers.ReshapeLayer(output, (-1, 38, 3))
        self.output = output
        reshape_output = lasagne.layers.ReshapeLayer(output, (-1, 38, 3))
        out = lasagne.layers.get_output(output)

        self._predict = theano.function([input_music_var, input_shift_var,
            input_position_var], out, allow_input_downcast=True)

        if trainable:
            weights = lasagne.layers.get_all_params(output, trainable=True)
            loss = lasagne.objectives.squared_error(out, delta_mov_var).mean()
            update = lasagne.updates.adam(loss, weights)

            self._train = theano.function([input_music_var, input_shift_var,
                input_position_var, delta_mov_var], loss, updates=update,
                allow_input_downcast=True)

    def predict(self, music, shifts, start_position):
        """
            music :2D tensor(time, chan)
            shifts :2D tensor(time, schifts)
            start_position :vector
            """
        return self._predict([music], [shifts], [start_position])

    def train(self, music, shifts, start_position, true_delta):
        """
            music :3D tensor(batch, time, chan)
            shifts :3D tensor(batch, time, schifts)
            start_position :matrix(batch, coord)
            """
        return self._train(music, shifts, start_position, true_delta)
        
    def save(self, file_name="weights.npy"):
        np.save(file_name, lasagne.layers.get_all_param_values(self.output))
        
    def load(self, file_name="weights.npy"):
        lasage.layers.set_all_param_values(self.output, np.load(file_name))
