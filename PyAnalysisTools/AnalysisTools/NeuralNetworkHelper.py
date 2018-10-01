import os
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, LSTM, Masking, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import plot_model
import keras.backend as backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from array import array
import ROOT
from MLHelper import Root2NumpyConverter, TrainingReader, DataScaler
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import make_dirs, copy
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as so
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
np.seterr(divide='ignore', invalid='ignore')


class LimitConfig(object):
    def __init__(self, name, **kwargs):
        kwargs.setdefault("nlayers", 3)
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
                    config[k] = v
            return config
        optimiser_type = optimiser_config.keys()[0]
        if optimiser_type == "sgd":
            return SGD(**convert_types(optimiser_config[optimiser_type]))
        if optimiser_type == "adagrad":
            return Adagrad(**convert_types(optimiser_config[optimiser_type]))
        if optimiser_type == "adam":
            return Adam(**convert_types(optimiser_config[optimiser_type]))


class NeuralNetwork(object):
    def __init__(self, num_features, limit_config):
        model = Sequential()
        width = 32
        model.add(Dense(units=width, input_dim=num_features, activation=limit_config.activation, kernel_initializer='random_normal'))
        for i in range(limit_config.nlayers - 1):
            model.add(Dense(width, activation=limit_config.activation, kernel_initializer='random_normal'))
            #model.add(Dropout(0.5))
        model.add(Dense(1, activation=limit_config.final_activation))
        model.compile(loss='binary_crossentropy', optimizer=limit_config.optimiser, metrics=['accuracy'])
        self.kerasmodel = model


class NNTrainer(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_features", None)
        kwargs.setdefault("units", 10)
        kwargs.setdefault("epochs", 10)
        kwargs.setdefault("control_plots", False)
        kwargs.setdefault("disable_scaling", False)
        kwargs.setdefault("verbosity", 1)
        self.reader = TrainingReader(**kwargs)
        self.variable_list = kwargs["variables"]
        self.converter = Root2NumpyConverter(self.variable_list + ["weight"])
        self.n_features = kwargs["n_features"]
        self.units = kwargs["units"]
        self.epochs = kwargs["epochs"]
        self.max_events = kwargs["max_events"]
        self.disable_rescaling = kwargs["disable_scaling"]
        self.plot = True
        self.do_control_plots = kwargs["control_plots"]
        self.output_path = self.get_resolved_output_path(kwargs["output_path"])
        if kwargs["icomb"] is not None:
            self.output_path = os.path.join(self.output_path, str(kwargs["icomb"]))
        self.limit_config = self.build_limit_config(kwargs["training_config_file"])
        self.selection = RegionBuilder(**yl.read_yaml(kwargs["selection_config"])["RegionBuilder"]).regions[0].event_cut_string
        self.store_arrays = not kwargs["disable_array_safe"]
        self.scaler = DataScaler(kwargs["scale_algo"])
        self.disable_event_weights = kwargs["disable_event_weights"]
        self.verbosity = kwargs["verbosity"]
        make_dirs(os.path.join(self.output_path, "plots"))
        make_dirs(os.path.join(self.output_path, "models"))
        make_dirs(os.path.join(self.output_path, "scalers"))
        copy(kwargs["training_config_file"], self.output_path)
        copy(kwargs["variable_set"], self.output_path)
        if self.store_arrays and not self.reader.numpy_input:
            self.input_store_path = os.path.join(self.output_path, "inputs")
            make_dirs(self.input_store_path)
        if self.reader.numpy_input:
            self.input_store_path = "/".join(kwargs["input_file"][0].split("/")[:-1])
        self.weight_train = None
        self.weight_eval = None

    @staticmethod
    def get_resolved_output_path(output_path):
        return so.resolve_output_dir(output_dir=output_path, sub_dir_name="NNtrain")

    @staticmethod
    def build_limit_config(config_file_name):
        configs = yl.read_yaml(config_file_name)
        for name, config in configs.iteritems():
            return LimitConfig(name, **config)

    def build_input(self):
        if not self.reader.numpy_input:
            trees = self.reader.get_trees()
            arrays = [[self.converter.convert_to_array(tree, max_events=self.max_events,
                                                       selection=self.selection[trees.index(items) % 2])
                       for tree in items] for items in trees]
            self.df_data_train, self.label_train = self.converter.merge(arrays[0], arrays[1])
            self.df_data_eval, self.label_eval = self.converter.merge(arrays[2], arrays[3])
            if self.store_arrays:
                np.save(os.path.join(self.input_store_path, "data_train.npy"), self.df_data_train)
                np.save(os.path.join(self.input_store_path, "label_train.npy"), self.label_train)
                np.save(os.path.join(self.input_store_path, "data_eval.npy"), self.df_data_eval)
                np.save(os.path.join(self.input_store_path, "label_eval.npy"), self.label_eval)
        else:
            self.df_data_train = np.load(os.path.join(self.input_store_path, "data_train.npy"))
            self.label_train = np.load(os.path.join(self.input_store_path, "label_train.npy"))
            self.df_data_eval = np.load(os.path.join(self.input_store_path, "data_eval.npy"))
            self.label_eval = np.load(os.path.join(self.input_store_path, "label_eval.npy"))
        self.df_data_train = pd.DataFrame(self.df_data_train)
        self.df_data_eval = pd.DataFrame(self.df_data_eval)

    def build_models(self):
        self.model_0 = NeuralNetwork(self.npa_data_train.shape[1], self.limit_config).kerasmodel
        self.model_1 = NeuralNetwork(self.npa_data_eval.shape[1], self.limit_config).kerasmodel

    def apply_scaling(self):
        self.npa_data_train, self.label_train = self.scaler.apply_scaling(self.npa_data_train, self.label_train,
                                                                          dump=os.path.join(self.output_path,
                                                                                            "scalers/train.pkl"))
        self.npa_data_eval, self.label_eval = self.scaler.apply_scaling(self.npa_data_eval, self.label_eval,
                                                                        dump=os.path.join(self.output_path,
                                                                                          "scalers/eval.pkl"))

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
        self.npa_data_train = self.df_data_train[self.variable_list].as_matrix()
        self.npa_data_eval = self.df_data_eval[self.variable_list].as_matrix()

        if not self.disable_event_weights:
            self.weight_train = self.df_data_train['weight']
            self.weight_eval = self.df_data_eval['weight']

        self.build_models()
        if self.do_control_plots:
            self.make_control_plots("prescaling")
        if not self.disable_rescaling:
            self.apply_scaling()
        if self.do_control_plots:
            self.make_control_plots("postscaling")

        history_train = self.model_0.fit(self.npa_data_train, self.label_train,
                                         epochs=self.epochs, verbose=self.verbosity, batch_size=64, shuffle=True,
                                         validation_data=(self.npa_data_eval, self.label_eval),
                                         sample_weight=self.weight_train)
        history_eval = self.model_1.fit(self.npa_data_eval, self.label_eval,
                                        epochs=self.epochs, verbose=self.verbosity, batch_size=64, shuffle=True,
                                        validation_data=(self.npa_data_train, self.label_train),
                                        sample_weight=self.weight_eval)
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
        preds_train = self.model_0.predict(self.npa_data_eval)
        preds_sig_train = preds_train[self.label_eval == 1]
        preds_bkg_train = preds_train[self.label_eval == 0]

        make_plot(preds_sig_train, preds_bkg_train, "train")
        preds_eval = self.model_1.predict(self.npa_data_train)
        preds_sig_eval = preds_eval[self.label_train == 1]
        preds_bkg_eval = preds_eval[self.label_train == 0]
        make_plot(preds_sig_eval, preds_bkg_eval, "eval")
        self.make_roc_curve(self.label_train, preds_eval, "train")
        self.make_roc_curve(self.label_eval, preds_train, "eval")

    def make_roc_curve(self, y, yhat, label):
        fpr, tpr, thresholds = roc_curve(y, yhat)

        roc_auc = auc(fpr, tpr)
        print "ROC AUC: %0.3f" % roc_auc
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='Full curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC curves for Signal vs Background')
        # plt.plot([0.038], [0.45], marker='*', color='red',markersize=5, label="Cut-based",linestyle="None")
        plt.plot([0.0, 0.0], [1., 1.], color='red', lw=1, linestyle='--')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_path, "plots/roc_{:s}.png".format(label)))
        plt.show()
        plt.clf()
        plt.close()

    def make_control_plots(self, prefix):
        def make_plot(prefix, variable_name, signal, background):
            data = signal
            data.append(background)
            if "/" in variable_name:
                variable_name = "_".join(variable_name.split("/")).replace(" ", "")
            var_range = np.percentile(data, [2.5, 97.5])
            plt.hist(map(float, signal.values), 100, range=var_range, histtype='step', label='signal', normed=True)
            plt.hist(map(float, background.values), 100, range=var_range, histtype='step', label='background',
                     normed=True)
            if data.ptp() > 1000.:
                plt.yscale('log')
            plt.legend(["signal", "background"], loc="upper right")
            plt.xlabel(variable_name)
            plt.ylabel('Normalised')

            plt.savefig(os.path.join(self.output_path, "plots/{:s}_{:s}.png".format(prefix, variable_name)))
            plt.close()

        df_data_train = pd.DataFrame(self.npa_data_train)
        df_data_eval = pd.DataFrame(self.npa_data_eval)
        for key, name in enumerate(self.variable_list):
            make_plot("{}_{}".format(prefix, "train"), name, df_data_train[key][self.label_train == 1],
                      df_data_train[key][self.label_train == 0])
            make_plot("{}_{}".format(prefix, "eval"), name, df_data_eval[key][self.label_eval == 1],
                      df_data_eval[key][self.label_eval == 0])
        if self.weight_train is not None:
            make_plot("{}_{}".format(prefix, "train"), "weight", self.weight_train[self.label_train == 1],
                      self.weight_train[self.label_train == 0])
        if self.weight_eval is not None:
            make_plot("{}_{}".format(prefix, "eval"), "weight", self.weight_eval[self.label_eval == 1],
                      self.weight_eval[self.label_eval == 0])


class NNReader(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("selection_config", None)
        self.file_handles = [FileHandle(file_name=fn, open_option="READ", run_dir=kwargs["run_dir"],
                                        dataset_info=kwargs["xs_config_file"])
                             for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.input_path = os.path.abspath(kwargs["input_path"])
        self.model_train = load_model(os.path.join(self.input_path, "models/model_train.h5"))
        self.model_eval = load_model(os.path.join(self.input_path, "models/model_eval.h5"))
        self.variable_list = kwargs["branches"]
        self.converter = Root2NumpyConverter(self.variable_list + ["weight", "train_flag"])
        self.converter_selection = Root2NumpyConverter(["event_number", "run_number"])
        self.branch_name = kwargs["branch_name"]
        self.friend_file_pattern = kwargs["friend_file_pattern"]
        self.friend_name = kwargs["friend_name"]
        self.input_path = os.path.abspath(kwargs["input_path"])
        self.output_path = kwargs["output_path"]
        self.selection = None
        if kwargs["selection_config"]:
            self.selection = RegionBuilder(**yl.read_yaml(kwargs["selection_config"])["RegionBuilder"]).regions[0].event_cut_string
        make_dirs(self.output_path)
        self.scaler = DataScaler(kwargs["scale_algo"])

    def build_friend_tree(self, file_handle):
        self.is_new_tree = False
        friend_file_name = file_handle.file_name.replace("hist",
                                                         self.friend_file_pattern).replace("ntuple",
                                                                                           self.friend_file_pattern)
        friend_file_name = os.path.join(self.output_path, friend_file_name.split("/")[-1])
        if not os.path.exists(friend_file_name):
            file_handle_friend = FileHandle(file_name=friend_file_name, open_option="RECREATE")
        else:
            file_handle_friend = FileHandle(file_name=friend_file_name, open_option="UPDATE")
        try:
            friend_tree = file_handle_friend.get_object_by_name(self.friend_name, "Nominal")
        except ValueError:
            self.is_new_tree = True
            friend_tree = ROOT.TTree(self.friend_name, "")
        return file_handle_friend, friend_tree

    def get_friend_tree(self, file_handle):
        friend_fh, friend_tree = self.build_friend_tree(file_handle)
        return friend_fh, friend_tree

    def run(self):
        for file_handle in self.file_handles:
            self.attach_NN_output(file_handle)

    def apply_scaling(self):
        self.npa_data_train, _ = self.scaler.apply_scaling(self.npa_data, None,
                                                           scaler=os.path.join(self.input_path, 'scalers/train.pkl'))
        self.npa_data_eval, _ = self.scaler.apply_scaling(self.npa_data, None,
                                                           scaler=os.path.join(self.input_path, 'scalers/eval.pkl'))

    def make_control_plots(self, train_pred, eval_pred, label):
        _logger.debug("Consistency plots")
        plt.hist(train_pred, 50, range=[0., 1.], histtype='step', label='model0', normed=True)
        plt.hist(eval_pred, 50, range=[0., 1.], histtype='step', label='model1', normed=True)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(["train", "eval"], loc="upper right")
        plt.savefig("plots/score_{:s}_log.png".format(label))
        plt.yscale('linear')
        plt.savefig("plots/score_{:s}_lin.png".format(label))
        plt.close()

    def make_variable_control_plots(self):
        def make_plot(prefix, variable_name, signal):
            data = signal
            if "/" in variable_name:
                variable_name = "_".join(variable_name.split("/")).replace(" ", "")
            var_range = np.percentile(data, [2.5, 97.5])
            plt.hist(map(float, signal), 100, range=var_range, histtype='step', label='signal', normed=False)
            if data.ptp() > 1000.:
                plt.yscale('log')
            plt.legend(["scaled"], loc="upper right")
            plt.xlabel(variable_name)
            plt.ylabel('Normalised')
            plt.savefig("plots/{:s}_{:s}.png".format(prefix, variable_name))
            plt.close()

        for key, name in enumerate(self.variable_list):
            make_plot("{}_{}".format("post_scaling", "train"), name, self.npa_data[:,key])

    def attach_NN_output(self, file_handle):
        tree = file_handle.get_object_by_name(self.tree_name, "Nominal")
        file_handle_friend, friend_tree = self.get_friend_tree(file_handle)
        selection = ""
        if self.selection is not None:
            selection = self.selection[0] if file_handle.is_mc else self.selection[1]

        data_selected = self.converter.convert_to_array(tree, selection)
        selected_event_numbers = pd.DataFrame(self.converter_selection.convert_to_array(tree, selection)).as_matrix()
        data_selected = pd.DataFrame(data_selected)

        self.npa_data = data_selected[self.variable_list].as_matrix()
        self.apply_scaling()
        prediction0 = self.model_train.predict(self.npa_data_train)
        prediction1 = self.model_eval.predict(self.npa_data_eval)
        self.make_variable_control_plots()
        nn_prediction = array('f', [0.])
        branch = friend_tree.Branch(self.branch_name, nn_prediction, "{:s}/F".format(self.branch_name))
        total_entries = tree.GetEntries()
        multiple_triplets = 0
        processed_events = 0
        already_processed_events = []
        for entry in range(total_entries):
            tree.GetEntry(entry)
            is_train = tree.train_flag == 0
            if not len(tree.object_pt) > 1 and (np.array([tree.event_number, tree.run_number]) == selected_event_numbers).all(1).any():
                processed_events += 1
                if (tree.event_number, tree.run_number) in already_processed_events:
                    _logger.error("ERROR already processed")
                try:
                    already_processed_events.append((tree.event_number, tree.run_number))
                    if not is_train:
                        nn_prediction[0] = prediction1[entry-multiple_triplets]
                    else:
                        nn_prediction[0] = prediction0[entry-multiple_triplets]
                except Exception as e:
                    print "exception ", processed_events
                    raise e
            else:
                nn_prediction[0] = -2
                multiple_triplets += 1
            if self.is_new_tree:
                friend_tree.Fill()
            branch.Fill()
        try:
            file_handle_friend.get_object_by_name("Nominal")
        except:
            file_handle_friend.tfile.mkdir("Nominal")
        tdir = file_handle_friend.get_directory("Nominal")
        tdir.cd()
        friend_tree.Write()
        file_handle_friend.close()
