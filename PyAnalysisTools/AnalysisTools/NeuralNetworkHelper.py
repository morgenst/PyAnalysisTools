import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, LSTM, Masking, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
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
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
np.seterr(divide='ignore', invalid='ignore')


class LimitConfig(object):
    def __init__(self, name, **kwargs):
        self.name = name
        for attr, value in kwargs.iteritems():
            if attr == "optimiser":
                setattr(self, "optimiser", self.set_optimiser(value))
                continue
            setattr(self, attr, value)

    @staticmethod
    def set_optimiser(optimiser_config):
        def convert_types(config):
            for k, v in config.iteritems():
                try:
                    config[k] = eval(v)
                except (NameError, TypeError):
                    pass
            return config
        optimiser_type = optimiser_config.keys()[0]
        if optimiser_type == "sgd":
            return SGD(**convert_types(optimiser_config[optimiser_type]))


class NeuralNetwork(object):
    def __init__(self, num_features, limit_config, num_layers=3):
        self.inputs = inputs = Input(shape=num_features)
        model = Sequential()
        model.add(Dense(64, input_dim=num_features[1], activation=limit_config.activation))
        for i in range(num_layers - 1):
            model.add(Dense(64, activation=limit_config.activation))
            model.add(Dropout(0.5))
        model.add(Dense(1, activation=limit_config.final_activation))
        model.compile(loss='binary_crossentropy', optimizer=limit_config.optimiser, metrics=['accuracy'])
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
        self.limit_config = self.build_limit_config(kwargs["training_config_file"])
        self.store_arrays = not kwargs["disable_array_safe"]
        make_dirs(os.path.join(self.output_path, "plots"))
        make_dirs(os.path.join(self.output_path, "models"))
        if self.store_arrays and not self.reader.numpy_input:
            self.input_store_path = os.path.join(self.output_path, "inputs")
            make_dirs(self.input_store_path)
        if self.reader.numpy_input:
            self.input_store_path = "/".join(kwargs["input_file"][0].split("/")[:-1])


    @staticmethod
    def build_limit_config(config_file_name):
        configs = YAMLLoader.read_yaml(config_file_name)
        for name, config in configs.iteritems():
            return LimitConfig(name, **config)

    def build_input(self):
        if not self.reader.numpy_input:
            trees = self.reader.get_trees()
            arrays = [[self.converter.convert_to_array(tree, max_events=self.max_events) for tree in items]
                      for items in trees]
            self.data_train, self.label_train = self.converter.merge(arrays[0], arrays[1])
            self.data_eval, self.label_eval = self.converter.merge(arrays[2], arrays[3])
            if self.store_arrays:
                np.save(os.path.join(self.input_store_path, "data_train.npy"), self.data_train)
                np.save(os.path.join(self.input_store_path, "label_train.npy"), self.label_train)
                np.save(os.path.join(self.input_store_path, "data_eval.npy"), self.data_eval)
                np.save(os.path.join(self.input_store_path, "label_eval.npy"), self.label_eval)
        else:
            self.data_train = np.load(os.path.join(self.input_store_path, "data_train.npy"))
            self.label_train = np.load(os.path.join(self.input_store_path, "label_train.npy"))
            self.data_eval = np.load(os.path.join(self.input_store_path, "data_eval.npy"))
            self.label_eval = np.load(os.path.join(self.input_store_path, "label_eval.npy"))
        self.data_train = pd.DataFrame(self.data_train)
        self.data_eval = pd.DataFrame(self.data_eval)

    def build_models(self):
        self.model_0 = NeuralNetwork(self.data_train.shape, self.limit_config, num_layers=self.layers).kerasmodel
        self.model_1 = NeuralNetwork(self.data_eval.shape, self.limit_config, num_layers=self.layers).kerasmodel

    def apply_scaling(self):
        train_mean_0 = self.data_train[self.label_train == 0].mean()
        train_mean_1 = self.data_train[self.label_train == 1].mean()
        train_std_0 = self.data_train[self.label_train == 0].std()
        train_std_1 = self.data_train[self.label_train == 1].std()
        self.data_train[self.label_train == 0] = (self.data_train[self.label_train == 0] - train_mean_0) / train_std_0
        self.data_train[self.label_train == 1] = (self.data_train[self.label_train == 1] - train_mean_1) / train_std_1
        eval_mean_0 = self.data_eval[self.label_eval == 0].mean()
        eval_mean_1 = self.data_eval[self.label_eval == 1].mean()
        eval_std_0 = self.data_eval[self.label_eval == 0].std()
        eval_std_1 = self.data_eval[self.label_eval == 1].std()
        self.data_eval[self.label_eval == 0] = (self.data_eval[self.label_eval == 0] - eval_mean_0) / eval_std_0
        self.data_eval[self.label_eval == 1] = (self.data_eval[self.label_eval == 1] - eval_mean_1) / eval_std_1
        
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
        history_train = self.model_0.fit(self.data_train.values, self.label_train.reshape((self.label_train.shape[0], 1)),
                                         epochs=self.epochs, verbose=1, batch_size=32, shuffle=True,
                                         validation_data=(self.data_eval.values, self.label_eval))
        history_eval = self.model_1.fit(self.data_eval.values, self.label_eval.reshape((self.label_eval.shape[0], 1)),
                                        epochs=self.epochs, verbose=1, batch_size=32, shuffle=True,
                                        validation_data=(self.data_train.values, self.label_train))
        if self.plot:
            self.plot_train_control(history_train, "train")
            self.plot_train_control(history_eval, "eval")
        _logger.debug("Done Training, saving models...")

        self.model_0.save(os.path.join(self.output_path, "models/model_train.h5"))
        self.model_1.save(os.path.join(self.output_path, "models/model_eval.h5"))
        self.run_predictions()
        self.plot_models()
        print "Stored outputs in {:s}".format(self.output_path)

    def plot_models(self):
        plot_model(self.model_0, to_file=os.path.join(self.output_path, "plots/model_train.png"), show_shapes=True)
        plot_model(self.model_1, to_file=os.path.join(self.output_path, "plots/model_eval.png"), show_shapes=True)

    def run_predictions(self):
        def make_plot(signal_pred, bkg_pred, label):
            _logger.debug("Consistency plots")
            sig_weights = np.ones_like(signal_pred) / float(len(signal_pred))
            bkg_weights = np.ones_like(bkg_pred) / float(len(bkg_pred))
            plt.hist(signal_pred, 50, range=[0., 1.], histtype='step', label='signal model0', weights=sig_weights,
                     normed=True)
            plt.hist(bkg_pred, 50, range=[0., 1.], histtype='step', label='bkg model1', weights=bkg_weights,
                     normed=True)
            plt.yscale('log')
            plt.grid(True)
            plt.legend(["signal", "background"], loc="upper right")
            plt.savefig(os.path.join(self.output_path, "plots/score_{:s}_log.png".format(label)))
            plt.yscale('linear')
            plt.savefig(os.path.join(self.output_path, "plots/score_{:s}_lin.png".format(label)))
            plt.close()

        if not self.plot:
            return
        _logger.info("Evaluating predictions")
        preds_train = self.model_0.predict(self.data_eval.values)
        preds_sig_train = preds_train[self.label_eval == 1]
        preds_bkg_train = preds_train[self.label_eval == 0]
        make_plot(preds_sig_train, preds_bkg_train, "train")
        preds_eval = self.model_1.predict(self.data_eval.values)
        preds_sig_eval = preds_eval[self.label_eval == 1]
        preds_bkg_eval = preds_eval[self.label_eval == 0]
        make_plot(preds_sig_eval, preds_bkg_eval, "eval")

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
