import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, LSTM, Masking, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from array import array
from MLHelper import Root2NumpyConverter, TrainingReader
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import make_dirs
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


class NeuralNetwork(object):
    def __init__(self, num_features, num_layers=3, size=20, lr=1e-3, keep_prob=1.0, tloss="soft", input_noise=0.0):
        self.inputs = inputs = Input(shape=(num_features,))
        x = Dense(size, activation='relu')(inputs)
        for i in range(num_layers - 1):
            x = Dense(size, activation='tanh')(x)
        pred = Dense(1, activation='sigmoid')(x)
        #pred = Dense(1, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=pred)

        # self.train_op = Adam(lr)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.kerasmodel = model


class NNTrainer(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_features", None)
        kwargs.setdefault("layers", 2)
        kwargs.setdefault("units", 10)
        kwargs.setdefault("epochs", 10)
        self.reader = TrainingReader(**kwargs)
        self.converter = Root2NumpyConverter(kwargs["variables"])
        self.n_features = kwargs["n_features"]
        self.layers = kwargs["layers"]
        self.units = kwargs["units"]
        self.epochs = kwargs["epochs"]
        self.plot = True
        self.output_path = kwargs["output_path"]
        make_dirs(os.path.join(self.output_path, "plots"))
        make_dirs(os.path.join(self.output_path, "models"))

    def build_input(self):
        trees = self.reader.get_trees()
        arrays = [[self.converter.convert_to_array(tree) for tree in items]
                  for items in trees]
        self.data_train, self.label_train = self.converter.merge(arrays[0], arrays[1])
        self.data_eval, self.label_eval = self.converter.merge(arrays[2], arrays[3])

    def build_models(self):
        if self.n_features is None:
            self.n_features = self.data_train.shape[1]
        self.model_0 = NeuralNetwork(self.n_features, num_layers=self.layers, size=self.units, tloss='soft').kerasmodel
        self.model_1 = NeuralNetwork(self.n_features, num_layers=self.layers, size=self.units, tloss='soft').kerasmodel

    def apply_scaling(self):
        train_mean = self.data_train.mean(0)
        train_std = self.data_train.std()
        self.data_train = (self.data_train - train_mean) / train_std
        eval_mean = self.data_eval.mean(0)
        eval_std = self.data_eval.std()
        self.data_eval = (self.data_eval - eval_mean) / eval_std

    def plot_train_control(self, history, name):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(["loss", "valid loss", "acc", "valid acc"])
        plt.xlabel('epoch')
        plt.ylabel('NN loss')
        plt.savefig(os.path.join(self.output_path, 'plots/{:s}.png'.format(name)))
        plt.close()

    def train(self):
        self.build_input()
        self.build_models()
        self.apply_scaling()
        history_train = self.model_0.fit(self.data_train, self.label_train, epochs=self.epochs, verbose=1, batch_size=32,
                                         shuffle=True, validation_data=(self.data_eval, self.label_eval))  # sample_weight=weight_0,
        history_eval = self.model_1.fit(self.data_eval, self.label_eval, epochs=self.epochs, verbose=1, batch_size=32,
                                        shuffle=True, validation_data=(self.data_train, self.label_train)) #sample_weight=weight_0,
        if self.plot:
            self.plot_train_control(history_train, "train")
            self.plot_train_control(history_eval, "eval")
        _logger.debug("Done Training, saving models...")

        self.model_0.save(os.path.join(self.output_path, "models/model_train.h5"))
        self.model_1.save(os.path.join(self.output_path, "models/model_eval.h5"))
        self.run_predictions()

    def run_predictions(self):
        _logger.info("Evaluating predictions")
        preds_train = self.model_0.predict(self.data_eval)
        preds_sig_train = preds_train[self.label_eval == 1]
        preds_bkg_train = preds_train[self.label_eval == 0]
        if self.plot:
            _logger.debug("Consistency plots")
            plt.hist(preds_sig_train, 20, range=[0., 1.], histtype='step', label='signal model0', normed=True)
            plt.hist(preds_bkg_train, 20, range=[0., 1.], histtype='step', label='bkg model1', normed=True)
            plt.yscale('log')
            plt.grid(True)
            plt.legend(["signal", "background"], loc="lower left")
            plt.savefig(os.path.join(self.output_path, "plots/consistency_sig_train.png"))


class NNReader(object):
    def __init__(self, **kwargs):
        self.file_handles = [FileHandle(file_name=fn, open_option="UPDATE", run_dir=kwargs["run_dir"])
                             for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.model_train = load_model(os.path.join(os.path.abspath(kwargs["model_path"]), "model_train.h5"))
        self.model_eval = load_model(os.path.join(os.path.abspath(kwargs["model_path"]), "model_eval.h5"))
        self.converter = Root2NumpyConverter(kwargs["branches"])
        self.branch_name = kwargs["branch_name"]

    def run(self):
        for file_handle in self.file_handles:
            self.attach_NN_output(file_handle)

    def attach_NN_output(self, file_handle):
        tree = file_handle.get_object_by_name(self.tree_name, "Nominal")
        data = self.converter.convert_to_array(tree)
        prediction0 = self.model_train.predict(data)
        prediction1 = self.model_eval.predict(data)
        bdt = array('f', [0.])
        branch = tree.Branch(self.branch_name, bdt, "{:s}/F".format(self.branch_name))
        total_entries = tree.GetEntries()
        multiple_triplets = 0
        for entry in range(total_entries):
            tree.GetEntry(entry)
            is_train = tree.train_flag == 0
            if not len(tree.object_pt) > 1:
                if not is_train:
                    bdt[0] = prediction1[entry-multiple_triplets]
                else:
                    bdt[0] = prediction0[entry-multiple_triplets]
            else:
                bdt[0] = -1
                multiple_triplets += 1
            branch.Fill()
        tdir = file_handle.get_directory("Nominal")
        tdir.cd()
        tree.Write()
