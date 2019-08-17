import numbers
import sys

from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as soh
try:
    import configManager
except ImportError:
    print "HistFitter not set up. Please run setup.sh in HistFitter directory. Giving up now."
    exit(1)


import ROOT
from ROOT import kBlack, kWhite, kGray, kRed, kPink, kMagenta, kViolet, kBlue, kAzure, kCyan, kTeal, kGreen, kSpring, \
    kYellow, kOrange, TCanvas, TLegend, TLegendEntry
from ROOT import *
from configWriter import fitConfig, Measurement, Channel, Sample
from systematic import Systematic
from math import sqrt
import os
from PyAnalysisTools.base.ShellUtils import make_dirs, copy, std_stream_redirected
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config, find_process_config, \
    transform_color
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


class ChannelDef(object):
    def __init__(self, name, config):
        self.name = name
        self.discr_var = config["discriminating_var"]["dist"].replace(" ", "")
        self.discr_var_xmin = config["discriminating_var"]["xmin"]
        self.discr_var_xmax = config["discriminating_var"]["xmax"]
        self.discr_var_bins = config["discriminating_var"]["bins"]
        self.cuts = " && ".join(config["cuts"])


class LimiConfig(object):
    def __init__(self, config_file):
        config = YAMLLoader().read_yaml(config_file)
        self.channels = []
        self.build_channel_configs(config)

    def build_channel_configs(self, config):
        for name, channel_config in config['channels'].iteritems():
            self.channels.append(ChannelDef(name, channel_config))


class HistFitterWrapper(object):
    @staticmethod
    def get_fit_modes():
        return ["bkg", "excl", "disc", "model-dep", "model-indep"]

    def prepare_output(self):
        make_dirs(os.path.join(self.output_dir, "results"))
        make_dirs(os.path.join(self.output_dir, "data"))
        make_dirs(os.path.join(self.output_dir, "config"))
        os.chdir(self.output_dir)
        copy(os.path.join(os.environ["HISTFITTER"], "config/HistFactorySchema.dtd"),
                          os.path.join(self.output_dir, "config/HistFactorySchema.dtd"))

    def clean(self):
        return
        if self.configMgr.executeHistFactory and not self.store_data:
            file_name = os.path.join(self.output_dir, "data", "{:s}.root".format(self.configMgr.analysisName))
            if os.path.isfile(file_name):
                os.remove(file_name)

    def __init__(self, **kwargs):
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("fit", False)
        kwargs.setdefault("fitname", "")
        kwargs.setdefault("minosPars", "")
        kwargs.setdefault("disable_limit_plot", True)
        kwargs.setdefault("hypotest", False)
        kwargs.setdefault("discovery_hypotest", False)
        kwargs.setdefault("draw", None)#"before,after,corrMatrix")
        kwargs.setdefault("draw_before", False)
        kwargs.setdefault("draw_after", False)
        kwargs.setdefault("drawCorrelationMatrix", False)
        kwargs.setdefault("drawSeparateComponents", False)
        kwargs.setdefault("drawLogLikelihood", False)
        kwargs.setdefault("drawSystematics", False)
        kwargs.setdefault("pickedSRs", [])
        kwargs.setdefault("runToys", False)
        kwargs.setdefault("no_empty", False)
        kwargs.setdefault("minos", False)
        kwargs.setdefault("minosPars", "")
        kwargs.setdefault("run_profiling", False)
        kwargs.setdefault("doFixParameters", False)
        kwargs.setdefault("fixedPars", "")
        kwargs.setdefault("validation", False)
        kwargs.setdefault("use_archive_histfile", False)
        kwargs.setdefault("read_tree", False)
        kwargs.setdefault("create_workspace", False)
        kwargs.setdefault("use_XML", True)
        kwargs.setdefault("num_toys", 1000)
        kwargs.setdefault("seed", 0)
        kwargs.setdefault("use_asimov", False)
        kwargs.setdefault("run_toys", False)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("base_output_dir", None)
        kwargs.setdefault("multi_core", False)
        kwargs.setdefault("store_data", False)

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                setattr(self, key, val)
        self.setup_output(**kwargs)
        self.samples = {}

    def __del__(self):
        del self.configMgr

    def setup_output(self, **kwargs):
        if not kwargs["multi_core"]:
            if not self.scan:
                self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
            elif self.scan:
                if self.base_output_dir is None:
                    self.base_output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
                self.output_dir = os.path.join(self.base_output_dir, str(self.call))
        else:
            self.output_dir = kwargs["output_dir"]
        self.prepare_output()

    def parse_configs(self):
        """
        Parse limit and process config files. Expands process config.
        :return: nothing
        :rtype: None
        """
        self.limit_config = None
        if hasattr(self, "limit_config_file"):
            self.limit_config = LimiConfig(self.limit_config_file)
        if self.process_config_file is not None:
            self.process_configs = parse_and_build_process_config(self.process_config_file)
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=os.path.abspath(self.xs_config_file)) for fn in self.input_files]

    def reset_config_mgr(self):
        try:
            import configManager
        except ImportError:
            print "HistFitter not set up. Please run setup.sh in HistFitter directory. Giving up now."
            exit(1)
        self.configMgr = configManager.configMgr
        self.configMgr.__init__()
        self.configMgr.analysisName = self.analysis_name
        self.configMgr.histCacheFile = os.path.join(self.output_dir, "data/{:s}.root".format(self.analysis_name))
        self.configMgr.outputFileName = os.path.join(self.output_dir,
                                                     "results/{:s}_Output.root".format(self.analysis_name))

    def initialise(self):
        if self.fit_mode == "bkg":
            self.configMgr.myFitType = self.configMgr.FitType.Background
            _logger.info("Will run in background-only fit mode")
        elif self.fit_mode == "excl" or self.fit_mode == "model-dep":
            self.configMgr.myFitType = self.configMgr.FitType.Exclusion
            _logger.info("Will run in exclusion (model-dependent) fit mode")
        elif self.fit_mode == "disc" or self.fit_mode == "model-indep":
            self.configMgr.myFitType = self.configMgr.FitType.Discovery
            _logger.info("Will run in discovery (model-independent) fit mode")
        else:
            _logger.error("fit type not specified. Giving up...")
            exit(0)

        if self.use_archive_histfile:
            self.configMgr.useHistBackupCacheFile = True

        if self.read_tree:
            self.configMgr.readFromTree = True

        if self.create_workspace:
            self.configMgr.executeHistFactory = True

        if self.use_XML:
            self.configMgr.writeXML = True

        #self.configMgr.userArg = self.userArg
        self.configMgr.nTOYs = self.num_toys

        # if self.log_level:
        #     _logger.setLevel(self.log_level, True)

        if self.hypotest:
            self.configMgr.doHypoTest = True

        if self.discovery_hypotest:
            self.configMgr.doDiscoveryHypoTest = True

        if self.no_empty:
            self.configMgr.removeEmptyBins = True

        if self.seed != 0:  # 0 is default because type is int
            self.configMgr.toySeedSet = True
            self.configMgr.toySeed = self.seed

        if self.use_asimov:
            self.configMgr.useAsimovSet = True

        if self.run_toys:
            runToys = True

        if self.minos:
            minosArgs = self.minos.split(",")
            for idx, arg in enumerate(minosArgs):
                if arg.lower() == "all":
                    minosArgs[idx] = "all"

            self.minosPars = ",".join(minosArgs)

        # if self.constant:
        #     doFixParameters = True
        #     fixedPars = self.constant

        gROOT.SetBatch(not self.interactive)

        """
        mandatory user-defined configuration file
        """
        """
        standard execution from now on
        """

        self.configMgr.initialize()
        RooRandom.randomGenerator().SetSeed(self.configMgr.toySeed)
        ReduceCorrMatrix = self.configMgr.ReduceCorrMatrix

        """
        runs Trees->histos and/or histos->workspace according to specifications
        """
        if self.configMgr.readFromTree or self.configMgr.executeHistFactory:
            if self.run_profiling:
                import cProfile
                cProfile.run('self.configMgr.executeAll()')
            else:
                self.configMgr.executeAll()

        """
        shows systematics
        """
        if self.drawSystematics:
            from ROOT import Util
            if not os.path.isdir("./plots"):
                _logger.info("no directory './plots' found - attempting to create one")
                os.mkdir("./plots")
            for fC in self.configMgr.fitConfigs:
                for chan in fC.channels:
                    for sam in chan.sampleList:
                        if not sam.isData:
                            if self.systematics:
                                Systs = self.systematics
                            else:
                                _logger.info("no systematic has been specified.... all the systematics will be considered")
                                Systs = ""
                                for i in sam.systDict.keys():
                                    Systs += i
                                    if 'Norm' in sam.systDict[i].method:
                                        Systs += "Norm"
                                    Systs += ","
                                if Systs != "":
                                    Systs = Systs[:-1]
                            if Systs != "":
                                Util.plotUpDown(self.configMgr.histCacheFile, sam.name, Systs,
                                                            chan.regionString, chan.variableName)

    # def build_samples(self):
    #     for fn in self.file_handles:
    #         merged_process = find_process_config(fn.process, self.process_configs)
    #         if merged_process is None:
    #             continue
    #         if merged_process.name in self.samples:
    #             self.samples[merged_process.name].files.append(fn.file_name)
    #         else:
    #             self.samples[merged_process.name] = Sample(merged_process.name, eval(merged_process.color))
    #             sample = self.samples[merged_process.name]
    #             if merged_process.type == "Data":
    #                 sample.setData()
    #             sample.setFileList([fn.file_name])
    #             sample.setTreeName("Nominal/BaseSelection_lq_tree_Final")

    def run_fit(self):
        """
        runs fitting and plotting, by calling C++ side functions
        """

        if self.fit or self.draw:
            idx = 0
            if len(self.configMgr.fitConfigs) == 0:
                _logger.fatal("No fit configurations found!")

            runAll = True
            if self.fitname != "":  # user specified a fit name
                fitFound = False
                for (i, config) in enumerate(self.configMgr.fitConfigs):
                    if self.configMgr.fitConfigs[i].name == self.fitname:
                        idx = i
                        fitFound = True
                        runAll = False
                        _logger.info("Found fitConfig with name %s at index %d" % (self.fitname, idx))
                        break

                if not fitFound:
                    _logger.fatal("Unable to find fitConfig with name %s, bailing out" % self.fitname)

            for i in xrange(len(self.configMgr.fitConfigs)):
                if not runAll and i != idx:
                    _logger.debug("Skipping fit config {0}".format(self.configMgr.fitConfigs[i].name))
                    continue

                _logger.info("Running on fitConfig %s" % self.configMgr.fitConfigs[i].name)
                _logger.info("Setting noFit = {0}".format(not self.fit))
                self.generate_fit_and_plot(self.configMgr.fitConfigs[i], self.configMgr.analysisName, self.draw_before,
                                           self.draw_after, self.drawCorrelationMatrix, self.drawSeparateComponents,
                                           self.drawLogLikelihood, self.minos, self.minosPars, self.doFixParameters,
                                           self.fixedPars, self.configMgr.ReduceCorrMatrix, not self.fit)
            _logger.debug(
                    "GenerateFitAndPlotCPP(self.configMgr.fitConfigs[%d], self.configMgr.analysisName, drawBeforeFit, "
                    "drawAfterFit, drawCorrelationMatrix, drawSeparateComponents, drawLogLikelihood, runMinos, "
                    "minosPars, doFixParameters, fixedPars, ReduceCorrMatrix)" % idx)
            _logger.debug(
                    "where drawBeforeFit, drawAfterFit, drawCorrelationMatrix, drawSeparateComponents, "
                    "drawLogLikelihood, ReduceCorrMatrix are booleans")

        """
        calculating and printing upper limits for model-(in)dependent signal fit configurations 
        (aka Exclusion/Discovery fit setup)
        """
        if not self.disable_limit_plot:
            for fc in self.configMgr.fitConfigs:
                if len(fc.validationChannels) > 0:
                    raise (Exception, "Validation regions should be turned off for setting an upper limit!")
                pass
            if not self.scan:
                self.configMgr.cppMgr.doUpperLimitAll()
            else:
                print self.call
                self.configMgr.cppMgr.doUpperLimit(self.call - 1)

        """
        run exclusion or discovery hypotest
        """
        if self.hypotest or self.discovery_hypotest:
            for fc in self.configMgr.fitConfigs:
                if len(fc.validationChannels) > 0 and not (fc.signalSample is None or 'Bkg' in fc.signalSample):
                    raise (Exception, "Validation regions should be turned off for doing hypothesis test!")
                pass

            if self.discovery_hypotest:
                self.configMgr.cppMgr.doHypoTestAll(os.path.join(self.output_dir, 'results/'), False)

            if self.hypotest:
                self.configMgr.cppMgr.doHypoTestAll(os.path.join(self.output_dir, 'results/'), True)

        if self.run_toys and self.configMgr.nTOYs > 0 and self.hypotest is False and self.disable_limit_plot and self.fit is False:
            self.configMgr.cppMgr.runToysAll()

        if self.interactive:
            from code import InteractiveConsole
            from ROOT import Util
            cons = InteractiveConsole(locals())
            cons.interact("Continuing interactive session... press Ctrl+d to exit")

        _logger.info("Leaving HistFitter... Bye!")

    def generate_fit_and_plot(self, fc, ana_name, draw_before_fit, draw_after_fit, draw_correlation_matrix,
                              draw_separate_components, draw_log_likelihood, minos, minos_pars, do_fix_parameters,
                              fixed_pars, reduce_corr_matrix, no_fit):
        """
        function call to top-level C++ side function Util.GenerateFitAndPlot()

        @param fc FitConfig name connected to fit and plot details
        @param ana_name Analysis name defined in config file, mainly used for output file/dir naming
        @param draw_before_fit Boolean deciding whether before-fit plots are produced
        @param draw_after_fit Boolean deciding whether after-fit plots are produced
        @param draw_correlation_matrix Boolean deciding whether correlation matrix plot is produced
        @param draw_separate_components Boolean deciding whether separate component (=sample) plots are produced
        @param draw_log_likelihood Boolean deciding whether log-likelihood plots are produced
        @param minos Boolean deciding whether asymmetric errors are calculated, eg whether MINOS is run
        @param minos_pars When minos is called, defining what parameters need asymmetric error calculation
        @param do_fix_parameters Boolean deciding if some parameters are fixed to a value given or not
        @param fixed_pars String of parameter1:value1,parameter2:value2 giving information on which parameter to fix to
        which value if dofixParameter == True
        @param reduce_corr_matrix Boolean deciding whether reduced correlation matrix plot is produced
        @param no_fit Don't re-run fit but use after-fit workspace
        """

        from ROOT import Util

        _logger.debug('GenerateFitAndPlotCPP: anaName %s ' % ana_name)
        _logger.debug("GenerateFitAndPlotCPP: drawBeforeFit %s " % draw_before_fit)
        _logger.debug("GenerateFitAndPlotCPP: drawAfterFit %s " % draw_after_fit)
        _logger.debug("GenerateFitAndPlotCPP: drawCorrelationMatrix %s " % draw_correlation_matrix)
        _logger.debug("GenerateFitAndPlotCPP: drawSeparateComponents %s " % draw_separate_components)
        _logger.debug("GenerateFitAndPlotCPP: drawLogLikelihood %s " % draw_log_likelihood)
        _logger.debug("GenerateFitAndPlotCPP: minos %s " % minos)
        _logger.debug("GenerateFitAndPlotCPP: minosPars %s " % minos_pars)
        _logger.debug("GenerateFitAndPlotCPP: doFixParameters %s " % do_fix_parameters)
        _logger.debug("GenerateFitAndPlotCPP: fixedPars %s " % fixed_pars)
        _logger.debug("GenerateFitAndPlotCPP: ReduceCorrMatrix %s " % reduce_corr_matrix)
        _logger.debug("GenerateFitAndPlotCPP: noFit {0}".format(no_fit))

        Util.GenerateFitAndPlot(fc.name, ana_name, draw_before_fit, draw_after_fit, draw_correlation_matrix,
                                draw_separate_components, draw_log_likelihood, minos, minos_pars, do_fix_parameters,
                                fixed_pars, reduce_corr_matrix, no_fit)


class HistFitterCountingExperiment(HistFitterWrapper):
    def __init__(self, **kwargs):
        kwargs.setdefault("bkg_name", "Bkg")
        kwargs.setdefault("analysis_name", "foo")
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        kwargs.setdefault("bkg_yields",  0.911)
        kwargs.setdefault("call", 0)
        kwargs.setdefault("scan", False)
        kwargs.setdefault("use_asimov", False)
        kwargs.setdefault("create_workspace", True)
        super(HistFitterCountingExperiment, self).__init__(**kwargs)
        self.control_regions = []
        self.validation_regions = []
        self.bkg_name = kwargs["bkg_name"]
        self.systematics = {}

    def run(self, **kwargs):
        self.control_regions = []
        if self.call > 0:
            self.setup_output(**kwargs)
        if False: #not kwargs["debug"] == True:
            with open(os.path.join(self.output_dir, "HistFitter.log"), 'w') as f, std_stream_redirected(f):
                with open(os.path.join(self.output_dir, "HistFitter.err"), 'w') as ferr, \
                        std_stream_redirected(ferr, sys.stderr):
                    self.setup_regions(**kwargs)
                    self.call += 1
                    self.run_fit()
        else:
            self.setup_regions(**kwargs)
            self.call += 1
            self.run_fit()

    @staticmethod
    def build_sample(name, yld, process_configs, region, sample=None):
        """
        Building a sample with single bin histo attached
        :param name: sample name
        :type name: string
        :param yld: event yield for process
        :type yld: float
        :param process_configs: process configuration containing style options
        :type process_configs: ProcessConfig
        :param region: region name for the build histogram, e.g. SR or CR
        :type region: string
        :param sample: (optional) already existing sample to which hist will be added
        :type sample: Sample
        :return: updated sample
        :rtype: Sample
        """
        if not isinstance(yld, numbers.Number):
            return
        if sample is None:
            sample = Sample(name, transform_color(process_configs[name].color))
        sample.setStatConfig(True)
        sample.buildHisto([yld], region, "yield", 0.5)
        sample.buildStatErrors([sqrt(yld)], region, "yield")
        sample.setNormByTheory()
        return sample

    def setup_single_background(self, **kwargs):
        nbkg_yields = kwargs["bkg_yields"]
        nbkg_err = sqrt(nbkg_yields)
        bkgSample = Sample(self.bkg_name, kGreen - 9)
        bkgSample.setStatConfig(True)
        bkgSample.buildHisto([nbkg_yields], "SR", self.var_name, 0.5)
        bkgSample.buildStatErrors([nbkg_err], "SR", self.var_name)
        return bkgSample

    def setup_multi_background(self, **kwargs):
        bkg_samples = []
        for bkg_name, bkg_yield in kwargs["bkg_yields"].iteritems():
            bkg_sample = self.build_sample(bkg_name, bkg_yield, kwargs["process_configs"], "SR")
            if not bkg_sample:
                continue
            bkg_samples.append(bkg_sample)
        return bkg_samples

    def setup_regions(self, **kwargs):
        kwargs.setdefault("sig_name", "Sig")
        kwargs.setdefault("sig_yield", 1.)
        kwargs.setdefault("control_regions", None)
        kwargs.setdefault("validation_regions", None)
        kwargs.setdefault("ctrl_config", None)
        kwargs.setdefault("var_name", "yield")
        kwargs.setdefault("sr_syst", None)
        kwargs.setdefault("fixed_signal", None)
        nbkg_yields = kwargs["bkg_yields"]
        self.var_name = kwargs["var_name"]

        self.reset_config_mgr()
        self.configMgr.cutsDict["SR"] = 1.
        self.configMgr.weights = "1."

        self.configMgr.doExclusion = True
        self.configMgr.nTOYs = 5000
        self.configMgr.calculatorType = 2
        self.configMgr.testStatType = 3
        self.configMgr.nPoints = 50

        self.configMgr.writeXML = True
        self.configMgr.blindSR = True
        self.configMgr.blindCR = False
        self.configMgr.blindVR = False

        samples = {}
        if isinstance(nbkg_yields, float):
            bkg_samples = [self.setup_single_background(**kwargs)]

        elif isinstance(nbkg_yields, dict):
            bkg_samples = self.setup_multi_background(**kwargs)

        if kwargs['fixed_signal'] is None:
            nsig = kwargs["sig_yield"]
        else:
            nsig = kwargs["fixed_signal"]

        nsig_err = 0.144
        lumi_error = 0.017

        dataSample = Sample("Data", kBlack)
        dataSample.setData()
        dataSample.buildHisto([5000.], "SR", self.var_name, 0.5)

        sigSample = Sample(kwargs["sig_name"], kPink)
        sigSample.setNormFactor("mu_Sig", 1., 0., 100.)
        sigSample.setStatConfig(True)
        sigSample.setNormByTheory()
        sigSample.buildHisto([nsig], "SR", self.var_name, 0.5)
        sigSample.buildStatErrors([nsig_err], "SR", self.var_name)

        for sample in bkg_samples + [sigSample, dataSample]:
            samples[sample.name] = sample
        self.samples = samples

        ana = self.configMgr.addFitConfig("SPlusB")
        ana.statErrorType = "Poisson"

        if kwargs["control_regions"] is not None:
            self.set_control_region_yields(ana=ana, samples=samples, **kwargs)
        # if kwargs["validation_regions"] is not None:
        #     self.setup_validation_regions(ana=ana, samples=samples, **kwargs)
        #     self.validation = True
        self.set_norm_factors(**kwargs)

        chan = ana.addChannel(self.var_name, ["SR"], 1, 0.5, 1.5)
        for sample in samples.values():
            chan.addSample(sample)
        ana.setSignalSample(sigSample)
        ana.addSignalChannels([chan])
        import logging
        logging.getLogger('fitConfig').setLevel(logging.DEBUG)
        if kwargs['control_regions'] is not None:
            for reg, yields in kwargs["control_regions"].iteritems():
                if 'ZVR' in reg:
                    continue
                reg_config = kwargs['ctrl_config'][reg]
                cr_chan = ana.addChannel(self.var_name, [reg], 1, 0.5, 1.5)
                for sample in self.samples.values():
                    cr_chan.addSample(sample)
                if reg_config['is_norm_region']:
                    ana.addBkgConstrainChannels([cr_chan])
                # elif reg_config['is_val_region']:
                #     ana.addValidationChannels([cr_chan])

        # print 'Signal channels: ', ana.signalChannels
        # print 'Background constrain channels: ', ana.bkgConstrainChannels
        # print 'Validation channels: ', ana.validationChannels
        # Define measurement
        meas = ana.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=lumi_error)
        meas.addPOI("mu_Sig")

        if kwargs['sr_syst'] is not None:
            systematics = kwargs['sr_syst']
            self.systematics["SR"] = {}
            for process in systematics.keys():
                if 'data' in process:
                    continue
                systs = set(map(lambda k: k.split("__")[0], systematics[process].keys()))
                uncert = systematics[process]
                if process not in self.systematics["SR"]:
                    self.systematics["SR"][process] = []
                for syst in systs:
                    try:
                        self.systematics["SR"][process].append(Systematic(name=syst.replace("weight", "alpha"),
                                                                          nominal=0., high=uncert[syst + '__1up'],
                                                                          low=uncert[syst + '__1down'],
                                                                          method='histoSys', type='user'))
                    except KeyError:
                        print "Could not find systematic {:s}".format(syst)
            #cr_chan = ana.getChannel(self.var_name, ['SR'])

        for cr in self.systematics.keys():
            if 'ZVR' in cr:
                continue
            cr_chan = ana.getChannel(self.var_name, [cr])
            for process, systematics in self.systematics[cr].iteritems():
                for syst in systematics:
                    cr_chan.getSample(process).addSystematic(syst)

        self.configMgr.cutsDict.keys()
        self.initialise()
        self.clean()

    def set_norm_factors(self, **kwargs):
        if kwargs['ctrl_config'] is None:
            return

        norm_factors = {}
        norm_regions = {}
        ctrl_config = kwargs["ctrl_config"]
        for region, config in ctrl_config.iteritems():
            if not config["is_norm_region"] and not config['is_val_region']:
                continue
            for bkg, bkg_config in config["bgk_to_normalise"].iteritems():
                if bkg not in self.samples:
                    _logger.error("Could not find background {:s} in samples".format(bkg))
                    continue
                if "norm_factor" in bkg_config:
                    norm_factor = bkg_config["norm_factor"]
                    try:
                        norm_factors[bkg].append(norm_factor)
                    except KeyError:
                        norm_factors[bkg] = [norm_factor]

                if config['is_norm_region']:
                    _logger.debug("Define norm region {:s} for background {:s}".format(region, bkg))
                    try:
                        norm_regions[bkg].append(region)
                    except KeyError:
                        norm_regions[bkg] = [region]
        for bkg, norm_factors in norm_factors.iteritems():
            for norm_factor in norm_factors:
                self.samples[bkg].setNormFactor(norm_factor, 1., 0.5, 1.5)

        for bkg, region in norm_regions.iteritems():
            _logger.debug("Set norm region {:s} and bkg {:s}".format(region, bkg))
            self.samples[bkg].setNormRegions([(region, self.var_name) for region in norm_regions[bkg]])

    def set_control_region_yields(self, **kwargs):
        data = kwargs["control_regions"]
        for reg, yields in data.iteritems():
            self.configMgr.cutsDict[reg] = 1.
            for process, yld in yields.iteritems():
                if process.lower() == "data" or process == kwargs['sig_name']:
                    continue
                sample = self.samples[process]
                sample.setStatConfig(True)
                sample.buildHisto([yld], reg, "yield", 0.5)
                sample.buildStatErrors([sqrt(yld)], reg, "yield")
            try:
                data_yld = filter(lambda kv: kv[0].lower() == "data", yields.iteritems())[0][1]
                if isinstance(data_yld, numbers.Number):
                    dataSample = self.samples['Data']
                    dataSample.setData()
                    dataSample.buildHisto([data_yld], reg, self.var_name, 0.5)
            except IndexError:
                _logger.error("No data found for region: {:s}".format(reg))
                exit()
            if kwargs['ctrl_syst'] is not None:
                systematics = kwargs['ctrl_syst'][reg]
                if reg not in self.systematics:
                    self.systematics[reg] = {}
                for process in systematics.keys():
                    systs = set(map(lambda k: k.split("__")[0], systematics[process].keys()))
                    uncert = systematics[process]
                    if process not in self.systematics[reg]:
                        self.systematics[reg][process] = []
                    for syst in systs:
                        try:
                            self.systematics[reg][process].append(Systematic(name=syst.replace("weight", "alpha"),
                                                                             nominal=0.,
                                                                             high=uncert[syst+'__1up'],
                                                                             low=uncert[syst+'__1down'],
                                                                             method='histoSys',
                                                                             type='user'))
                        except KeyError:
                            _logger.error("Could not find systematic {:s}".format(syst))

    def get_upper_limit(self, name="hypo_Sig"):
        f = ROOT.TFile.Open(os.path.join(self.output_dir,
                                         "results/{:s}_Output_upperlimit.root".format(self.configMgr.analysisName)),
                            "READ")
        result = f.Get(name)
        try:
            return result.GetExpectedUpperLimit(), result.GetExpectedUpperLimit(1), result.GetExpectedUpperLimit(-1)
        except AttributeError:
            return -1., 0., 0.


# class HistFitterShapeAnalysis(HistFitterWrapper):
#     def __init__(self, **kwargs):
#         kwargs.setdefault("name", "ShapeAnalysis")
#         kwargs.setdefault("read_tree", True)
#         kwargs.setdefault("create_workspace", True)
#         kwargs.setdefault("output_dir", kwargs["output_dir"])
#         super(HistFitterShapeAnalysis, self).__init__(**kwargs)
#         self.parse_configs()
#         self.configMgr.calculatorType = 2
#         self.configMgr.testStatType = 3
#         self.configMgr.nPoints = 20
#         FitType = self.configMgr.FitType
#         self.configMgr.writeXML = True
#         self.analysis_name = kwargs["name"]
#
#         self.configMgr.blindSR = True
#         self.configMgr.blindCR = False
#         self.configMgr.blindVR = False
#         # self.configMgr.useSignalInBlindedData = True
#         cur_dir = os.path.abspath(os.path.curdir)
#
#         # First define HistFactory attributes
#         self.configMgr.analysisName = self.analysis_name
#
#         # Scaling calculated by outputLumi / inputLumi
#         self.configMgr.inputLumi = 100  # Luminosity of input TTree after weighting
#         self.configMgr.outputLumi = 100. #4.713  # Luminosity required for output histograms
#         self.configMgr.setLumiUnits("fb-1")
#
#         for channel in self.limit_config.channels:
#             self.configMgr.cutsDict[channel.name] = channel.cuts
#         self.configMgr.cutsDict["SR"] = "(electron_pt > 65000.)"
#         self.configMgr.weights = ["weight"]
#
#         self.build_samples()
#
#         # **************
#         # Exclusion fit
#         # **************
#         if True: #myFitType == FitType.Exclusion:
#
#             # loop over all signal points
#             # Fit config instance
#             exclusionFitConfig = self.configMgr.addFitConfig("Exclusion_LQ")
#             meas = exclusionFitConfig.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=0.039)
#             meas.addPOI("mu_SIG")
#
#             exclusionFitConfig.addSamples(self.samples.values())
#             # Systematics
#             sigSample = Sample("LQ", kPink)
#             sigSample.setFileList(["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"])
#             sigSample.setTreeName("Nominal/BaseSelection_lq_tree_Final")
#             #sigSample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)
#             sigSample.setNormByTheory()
#             sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
#             # sigSample.addSampleSpecificWeight("0.001")
#             exclusionFitConfig.addSamples(sigSample)
#             exclusionFitConfig.setSignalSample(sigSample)
#             regions = []
#             for channel in self.limit_config.channels:
#                 region = exclusionFitConfig.addChannel(channel.discr_var, [channel.name], channel.discr_var_bins,
#                                                        channel.discr_var_xmin, channel.discr_var_xmax)
#                 region.useOverflowBin = True
#                 region.useUnderflowBin = True
#                 regions.append(region)
#             #exclusionFitConfig.addSignalChannels([srBin])
#
#             exclusionFitConfig.addSignalChannels(regions)
#         self.initialise()
#         self.FitType = self.configMgr.FitType
#         # First define HistFactory attributes
#         # Scaling calculated by outputLumi / inputLumi
#         self.configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
#         self.configMgr.outputLumi = 4.713  # Luminosity required for output histograms
#         self.configMgr.setLumiUnits("fb-1")
#         self.configMgr.calculatorType = 2
#         #self.configMgr.histCacheFile = "data/" + self.configMgr.analysisName + ".root"
#
#         useStat = True
#         # Tuples of nominal weights without and with b-jet selection
#         self.configMgr.weights = ("weight")
#
#         # name of nominal histogram for systematics
#         self.configMgr.nomName = "_NoSys"
#
#
#         # -----------------------------
#         # Exclusion fits (1-step simplified model in this case)
#         # -----------------------------
#         doValidation = False
#         # if True: #myFitType == FitType.Exclusion:
#         #     sigSamples = ["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"]
#         #     #self.dataSample.buildHisto([1., 6., 16., 3., 0.], "SS", "lq_mass_max", 0.2, 0.1)
#         #
#         #     for sig in sigSamples:
#         #         #myTopLvl = self.configMgr.addFitConfigClone(bkt, "Sig_%s" % sig)
#         #         sigSample = Sample(sig, kPink)
#         #         sigSample.setFileList([sig])
#         #         sigSample.setNormByTheory()
#         #         sigSample.setStatConfig(useStat)
#         #         sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
#         #         # bkt.addSamples(sigSample)
#         #         # bkt.setSignalSample(sigSample)
#         #
#         #         # s1l2j using met/meff
#         #         # if doValidation:
#         #         #     mm2J = myTopLvl.getChannel("met/meff2Jet", ["SS"])
#         #         #     iPop = myTopLvl.validationChannels.index("SS_metmeff2Jet")
#         #         #     myTopLvl.validationChannels.pop(iPop)
#         #         # else:
#         #         #     mm2J = myTopLvl.addChannel("met/meff2Jet", ["SS"], 5, 0.2, 0.7)
#         #         #     mm2J.useOverflowBin = True
#         #         #     mm2J.addSystematic(jes)
#         #         #     pass
#         #         #myTopLvl.addSignalChannels([mm2J])
#         #
