import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, LSTM, Masking, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

import keras.backend as backend
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from array import array
from MLHelper import Root2NumpyConverter, TrainingReader
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import make_dirs
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as so
np.seterr(divide='ignore', invalid='ignore')


class NeuralNetwork(object):
    def __init__(self, num_features, num_layers=3, size=20, lr=1e-3, keep_prob=1.0, tloss="soft", input_noise=0.0):
        # self.inputs = inputs = Input(shape=num_features)
        # print backend.int_shape(inputs)
        # x = Reshape((-1,))(inputs)
        # print backend.int_shape(x)
        #
        # x = Dense(size, activation='relu')(x)
        # for i in range(num_layers - 1):
        #     x = Dense(size, activation='tanh')(x)
        # #pred = Dense(1, activation='softmax')(x)
        # pred = Dense(1, activation="sigmoid")(x)
        # model = Model(inputs=inputs, outputs=pred)
        # # self.train_op = Adam(lr)
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.kerasmodel = model

        self.inputs = inputs = Input(shape=num_features)
        # print backend.int_shape(inputs)
        # x = Reshape((-1,))(inputs)
        # print backend.int_shape(x)
        model =Sequential()
        #model.add(Dense(num_features[0], input_shape=(num_features[1],)))
        model.add(Dense(64, input_dim=num_features[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        # model.add(Activation('relu'))
        # x = Dense(size, activation='relu')(x)
        # for i in range(num_layers - 1):
        #     x = Dense(size, activation='tanh')(x)
        # #pred = Dense(1, activation='softmax')(x)
        # pred = Dense(1, activation="sigmoid")(x)
        # model = Model(inputs=inputs, outputs=pred)
        # self.train_op = Adam(lr)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.kerasmodel = model


class NNTrainer(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_features", None)
        kwargs.setdefault("layers", 2)
        kwargs.setdefault("units", 10)
        kwargs.setdefault("epochs", 10)
        kwargs.setdefault("control_plots", False)
        self.reader = TrainingReader(**kwargs)
        self.converter = Root2NumpyConverter(kwargs["variables"])
        self.n_features = kwargs["n_features"]
        self.layers = kwargs["layers"]
        self.units = kwargs["units"]
        self.epochs = kwargs["epochs"]
        self.max_events = kwargs["max_events"]
        self.plot = True
        self.do_control_plots = kwargs["control_plots"]
        self.output_path = so.resolve_output_dir(output_dir=kwargs["output_path"], sub_dir_name="NNtrain")
        make_dirs(os.path.join(self.output_path, "plots"))
        make_dirs(os.path.join(self.output_path, "models"))

    def build_input(self):
        trees = self.reader.get_trees()
        arrays = [[self.converter.convert_to_array(tree, max_events=self.max_events) for tree in items]
                  for items in trees]
        self.data_train, self.label_train = self.converter.merge(arrays[0], arrays[1])
        self.data_eval, self.label_eval = self.converter.merge(arrays[2], arrays[3])
        self.data_train = pd.DataFrame(self.data_train)
        self.data_eval = pd.DataFrame(self.data_eval)

    def build_models(self):
        self.model_0 = NeuralNetwork(self.data_train.shape, num_layers=self.layers, size=self.units,
                                     tloss='soft').kerasmodel
        self.model_1 = NeuralNetwork(self.data_eval.shape, num_layers=self.layers, size=self.units,
                                     tloss='soft').kerasmodel

    def apply_scaling(self):
        train_mean = self.data_train.mean()
        train_std = self.data_train.std()
        self.data_train = (self.data_train - train_mean) / train_std
        eval_mean = self.data_eval.mean()
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
        if self.do_control_plots:
            self.make_control_plots("prescaling")
        self.apply_scaling()
        if self.do_control_plots:
            self.make_control_plots("postscaling")
        #print "train shape: ", train_shape
        history_train = self.model_0.fit(self.data_train.values, self.label_train.reshape((self.label_train.shape[0], 1)),
                                         epochs=self.epochs, verbose=1, batch_size=32,
                                         shuffle=True, validation_data=(self.data_eval.values, self.label_eval))  # sample_weight=weight_0,
        history_eval = self.model_1.fit(self.data_eval.values, self.label_eval.reshape((self.label_eval.shape[0], 1)), epochs=self.epochs,
                                        verbose=1, batch_size=32,
                                        shuffle=True, validation_data=(self.data_train.values, self.label_train)) #sample_weight=weight_0,
        if self.plot:
            self.plot_train_control(history_train, "train")
            self.plot_train_control(history_eval, "eval")
        _logger.debug("Done Training, saving models...")

        self.model_0.save(os.path.join(self.output_path, "models/model_train.h5"))
        self.model_1.save(os.path.join(self.output_path, "models/model_eval.h5"))
        self.run_predictions()

    def run_predictions(self):
        _logger.info("Evaluating predictions")
        preds_train = self.model_0.predict(self.data_eval.values)
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

    def make_control_plots(self, prefix):
        def make_plot(prefix, variable_name, signal, background):
            data = signal
            data.append(background)
            if "/" in variable_name:
                variable_name = "_".join(variable_name.split("/")).replace(" ","")
            var_range = np.percentile(data, [2.5, 97.5])
            plt.hist(map(float, signal.values), 100, range=var_range, histtype='step', label='signal', normed=True)
            plt.hist(map(float, background.values), 100, range=var_range, histtype='step', label='background', normed=True)
            if data.ptp() > 1000.:
                plt.yscale('log')
            plt.legend(["signal", "background"], loc="upper right")
            plt.xlabel(variable_name)
            plt.ylabel('Normalised')

            plt.savefig(os.path.join(self.output_path, "plots/{:s}_{:s}.png".format(prefix, variable_name)))
            plt.close()

        for key in self.data_train.keys():
            make_plot("{}_{}".format(prefix, "train"), key, self.data_train[key][self.label_train == 1],
                      self.data_train[key][self.label_train == 0])
            make_plot("{}_{}".format(prefix, "eval"), key, self.data_eval[key][self.label_eval == 1],
                      self.data_eval[key][self.label_eval == 0])


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
