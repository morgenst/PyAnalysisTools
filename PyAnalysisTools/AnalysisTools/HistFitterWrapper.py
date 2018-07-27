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

    def __init__(self, **kwargs):
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("fit", True)
        kwargs.setdefault("fitname", "")
        kwargs.setdefault("minosPars", "")
        kwargs.setdefault("disable_limit_plot", False)
        kwargs.setdefault("hypotest", False)
        kwargs.setdefault("discovery_hypotest", False)
        kwargs.setdefault("draw", "before,after,corrMatrix")
        kwargs.setdefault("draw_before", False)
        kwargs.setdefault("draw_after", False)
        kwargs.setdefault("drawCorrelationMatrix", True)
        kwargs.setdefault("drawSeparateComponents", False)
        kwargs.setdefault("drawLogLikelihood", True)
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
        kwargs.setdefault("read_tree", False) #self.configMgr.readFromTree
        kwargs.setdefault("create_workspace", False) #self.configMgr.executeHistFactory)
        #kwargs.setdefault("use_XML", self.configMgr.writeXML)
        kwargs.setdefault("use_XML", True)
        kwargs.setdefault("num_toys", 1000) #self.configMgr.nTOYs)
        kwargs.setdefault("seed", 0) #self.configMgr.toySeed)
        kwargs.setdefault("use_asimov", False) #self.configMgr.useAsimovSet)
        kwargs.setdefault("run_toys", False)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("base_output_dir", None)

        #FitType = self.configMgr.FitType  # enum('FitType','Discovery , Exclusion , Background')
        #myFitType = FitType.Background

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                setattr(self, key, val)
        self.setup_output(**kwargs)
        self.samples = {}

    def __del__(self):
        del self.configMgr

    def setup_output(self, **kwargs):
        if not self.scan:
            self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
        elif self.scan:
            if self.base_output_dir is None:
                self.base_output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
            self.output_dir = os.path.join(self.base_output_dir, str(self.call))
        self.prepare_output()

    def parse_configs(self):
        self.limit_config = None
        if hasattr(self, "limit_config_file"):
            self.limit_config = LimiConfig(kwargs["limit_config_file"])
        if self.process_config_file is not None:
            self.process_configs = parse_and_build_process_config(self.process_config_file)
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=os.path.abspath(self.xs_config_file)) for fn in self.input_files]
        self.expand_process_configs()

    def reset_config_mgr(self):
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

    def expand_process_configs(self):
        if self.process_configs is not None:
            for fh in self.file_handles:
                    _ = find_process_config(fh.process, self.process_configs)

    def build_samples(self):
        for fn in self.file_handles:
            merged_process = find_process_config(fn.process, self.process_configs)
            if merged_process is None:
                continue
            if merged_process.name in self.samples:
                self.samples[merged_process.name].files.append(fn.file_name)
            else:
                self.samples[merged_process.name] = Sample(merged_process.name, eval(merged_process.color))
                sample = self.samples[merged_process.name]
                if merged_process.type == "Data":
                    sample.setData()
                sample.setFileList([fn.file_name])
                sample.setTreeName("Nominal/BaseSelection_lq_tree_Final")
                #sample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)

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

        # draw_after_fit = False
        # draw_before_fit = False
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
        self.bkg_name = kwargs["bkg_name"]

    def run(self, **kwargs):
        self.control_regions = []
        if self.call > 0:
            self.setup_output(**kwargs)

        with open(os.path.join(self.output_dir, "HistFitter.log"), 'w') as f, std_stream_redirected(f):
            with open(os.path.join(self.output_dir, "HistFitter.err"), 'w') as ferr, \
                    std_stream_redirected(ferr, sys.stderr):
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
        return sample

    def setup_single_background(self, **kwargs):
        nbkg_yields = kwargs["bkg_yields"]
        nbkg_err = sqrt(nbkg_yields)
        bkgSample = Sample(self.bkg_name, kGreen - 9)
        bkgSample.setStatConfig(True)
        bkgSample.buildHisto([nbkg_yields], "SR", kwargs["yield"], 0.5)
        bkgSample.buildStatErrors([nbkg_err], "SR", kwargs["var_name"])
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
        kwargs.setdefault("ctrl_config", None)
        kwargs.setdefault("var_name", "yield")
        nbkg_yields = kwargs["bkg_yields"]
        var_name = kwargs["var_name"]

        self.reset_config_mgr()
        self.configMgr.cutsDict["SR"] = 1.
        self.configMgr.weights = "1."

        self.configMgr.doExclusion = False  # True=exclusion, False=discovery
        self.configMgr.nTOYs = 5000
        self.configMgr.calculatorType = 2  # 2=asymptotic calculator, 0=frequentist calculator
        self.configMgr.testStatType = 3  # 3=one-sided profile likelihood test statistic (LHC default)
        self.configMgr.nPoints = 50  # number of values scanned of signal-strength for upper-limit determination of signal strength.

        self.configMgr.writeXML = True
        self.configMgr.blindSR = True
        self.configMgr.blindCR = False
        self.configMgr.blindVR = False

        samples = {}
        if isinstance(nbkg_yields, float):
            bkg_samples = [self.setup_single_background(**kwargs)]
            ndata = nbkg_yields

        elif isinstance(nbkg_yields, dict):
            bkg_samples = self.setup_multi_background(**kwargs)
            ndata = 0

        nsig = kwargs["sig_yield"]  # Number of predicted signal events
        nsig_err = 0.144  # (Absolute) Statistical error on signal estimate
        lumi_error = 0.039  # Relative luminosity uncertainty

        dataSample = Sample("Data", kBlack)
        dataSample.setData()
        dataSample.buildHisto([5.], "SR", var_name, 0.5) #ndata

        sigSample = Sample(kwargs["sig_name"], kPink)
        sigSample.setNormFactor("mu_Sig", 1., 0., 100.)
        sigSample.setStatConfig(True)
        sigSample.setNormByTheory()
        sigSample.buildHisto([nsig], "SR", var_name, 0.5)
        sigSample.buildStatErrors([nsig_err], "SR", var_name)
        for sample in bkg_samples + [sigSample, dataSample]:
            samples[sample.name] = sample
        ana = self.configMgr.addFitConfig("SPlusB")

        # bkgSample.addSystematic(corb)
        # bkgSample.addSystematic(ucb)
        if kwargs["control_regions"] is not None:
            self.setup_control_regions(ana=ana, samples=samples, **kwargs)
            if kwargs["ctrl_config"] is not None:
                norm_factors = {}
                norm_regions = {}
                ctrl_config = kwargs["ctrl_config"]
                for region, config in ctrl_config.iteritems():
                    if not config["is_norm_region"]:
                        continue
                    for bkg, bkg_config in config["bgk_to_normalise"].iteritems():
                        if bkg not in samples:
                            print "ERROR: Could not find background {:s} in samples".format(bkg)
                            continue
                        try:
                            norm_regions[bkg].append(region)
                        except KeyError:
                            norm_regions[bkg] = [region]
                        if "norm_factor" in bkg_config:
                            norm_factor = bkg_config["norm_factor"]
                            try:
                                norm_factors[bkg].append(norm_factor)
                            except KeyError:
                                norm_factors[bkg] = [norm_factor]
                for bkg, norm_factors in norm_factors.iteritems():
                    for norm_factor in norm_factors:
                        samples[bkg].setNormFactor(norm_factor, 1., 0., 5.)
                for bkg in norm_regions.keys():
                    samples[bkg].setNormRegions([(region, var_name) for region in norm_regions[bkg]])

        ana.addSamples(samples.values())
        ana.setSignalSample(sigSample)

        # Define measurement
        meas = ana.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=lumi_error)
        meas.addPOI("mu_Sig")
        # meas.addParamSetting("Lumi",True,1)

        chan = ana.addChannel(var_name, ["SR"], 1, 0.5, 1.5)
        ana.addSignalChannels([chan])
        cr_channels = []
        for cr in self.control_regions:
            cr_channels.append(ana.addChannel(var_name, [cr], 1, 0.5, 1.5))
        if len(cr_channels) > 0:
            ana.addBkgConstrainChannels(cr_channels)
        self.configMgr.cutsDict.keys()
        self.initialise()

        if self.configMgr.executeHistFactory:
            file_name = os.path.join(self.output_dir, "data", "{:s}.root".format(self.configMgr.analysisName))
            if os.path.isfile(file_name):
                os.remove(file_name)

    def setup_control_regions(self, **kwargs):
        data = kwargs["control_regions"]
        samples = kwargs["samples"]
        cr_channels = []
        ana = kwargs["ana"]
        for reg, yields in data.iteritems():
            self.configMgr.cutsDict[reg] = 1.
            for process, yld in yields.iteritems():
                if process.lower() == "data":
                    continue
                sample = None
                try:
                    _ = self.build_sample(process, yld[0], kwargs["process_configs"], reg, samples[process])
                except Exception:
                    sample = self.build_sample(process, yld[0], kwargs["process_configs"], reg)
                    if not sample:
                        print "could not find sample ", process
                        continue
                if sample is not None:
                    samples[sample.name] = sample
            try:
                data_yld = filter(lambda kv: kv[0].lower() == "data", yields.iteritems())[0][1][0]
                if isinstance(data_yld, numbers.Number):
                    dataSample = samples["Data"]
                    dataSample.buildHisto([data_yld], reg, kwargs["var_name"], 0.5)
            except IndexError:
                print "No data found for ", reg
            self.control_regions.append(reg)

    def get_upper_limit(self, name="hypo_Sig"):
        f = ROOT.TFile.Open(os.path.join(self.output_dir,
                                         "results/{:s}_Output_upperlimit.root".format(self.configMgr.analysisName)),
                            "READ")
        result = f.Get(name)
        try:
            return result.GetExpectedUpperLimit(), result.GetExpectedUpperLimit(1), result.GetExpectedUpperLimit(-1)
        except AttributeError:
            return -1., 0., 0.


class HistFitterShapeAnalysis(HistFitterWrapper):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ShapeAnalysis")
        kwargs.setdefault("read_tree", True)
        kwargs.setdefault("create_workspace", True)
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        super(HistFitterShapeAnalysis, self).__init__(**kwargs)
        self.parse_configs()
        self.configMgr.calculatorType = 2
        self.configMgr.testStatType = 3
        self.configMgr.nPoints = 20
        FitType = self.configMgr.FitType
        self.configMgr.writeXML = True
        self.analysis_name = kwargs["name"]

        self.configMgr.blindSR = True
        self.configMgr.blindCR = False
        self.configMgr.blindVR = False
        # self.configMgr.useSignalInBlindedData = True
        cur_dir = os.path.abspath(os.path.curdir)

        # First define HistFactory attributes
        self.configMgr.analysisName = self.analysis_name

        # Scaling calculated by outputLumi / inputLumi
        self.configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        self.configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        self.configMgr.setLumiUnits("fb-1")

        for channel in self.limit_config.channels:
            self.configMgr.cutsDict[channel.name] = channel.cuts
        self.configMgr.cutsDict["SR"] = "(electron_pt > 65000.)"
        self.configMgr.weights = ["weight"]

        self.build_samples()

        # **************
        # Exclusion fit
        # **************
        if True: #myFitType == FitType.Exclusion:

            # loop over all signal points
            # Fit config instance
            exclusionFitConfig = self.configMgr.addFitConfig("Exclusion_LQ")
            meas = exclusionFitConfig.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=0.039)
            meas.addPOI("mu_SIG")

            exclusionFitConfig.addSamples(self.samples.values())
            # Systematics
            #exclusionFitConfig.getSample("Top").addSystematic(topKtScale)
            #exclusionFitConfig.getSample("WZ").addSystematic(wzKtScale)
            #exclusionFitConfig.addSystematic(jes)
            sigSample = Sample("LQ", kPink)
            sigSample.setFileList(["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"])
            sigSample.setTreeName("Nominal/BaseSelection_lq_tree_Final")
            #sigSample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)
            sigSample.setNormByTheory()
            sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
            # sigSample.addSampleSpecificWeight("0.001")
            exclusionFitConfig.addSamples(sigSample)
            exclusionFitConfig.setSignalSample(sigSample)
            regions = []
            for channel in self.limit_config.channels:
                region = exclusionFitConfig.addChannel(channel.discr_var, [channel.name], channel.discr_var_bins,
                                                       channel.discr_var_xmin, channel.discr_var_xmax)
                region.useOverflowBin = True
                region.useUnderflowBin = True
                regions.append(region)
            #exclusionFitConfig.addSignalChannels([srBin])

            exclusionFitConfig.addSignalChannels(regions)
        self.initialise()
        self.FitType = self.configMgr.FitType
        # First define HistFactory attributes
        # Scaling calculated by outputLumi / inputLumi
        self.configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        self.configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        self.configMgr.setLumiUnits("fb-1")
        self.configMgr.calculatorType = 2
        #self.configMgr.histCacheFile = "data/" + self.configMgr.analysisName + ".root"

        useStat = True
        # Tuples of nominal weights without and with b-jet selection
        self.configMgr.weights = ("weight")

        # name of nominal histogram for systematics
        self.configMgr.nomName = "_NoSys"





        # -----------------------------
        # Exclusion fits (1-step simplified model in this case)
        # -----------------------------
        doValidation = False
        # if True: #myFitType == FitType.Exclusion:
        #     sigSamples = ["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"]
        #     #self.dataSample.buildHisto([1., 6., 16., 3., 0.], "SS", "lq_mass_max", 0.2, 0.1)
        #
        #     for sig in sigSamples:
        #         #myTopLvl = self.configMgr.addFitConfigClone(bkt, "Sig_%s" % sig)
        #         sigSample = Sample(sig, kPink)
        #         sigSample.setFileList([sig])
        #         sigSample.setNormByTheory()
        #         sigSample.setStatConfig(useStat)
        #         sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
        #         # bkt.addSamples(sigSample)
        #         # bkt.setSignalSample(sigSample)
        #
        #         # s1l2j using met/meff
        #         # if doValidation:
        #         #     mm2J = myTopLvl.getChannel("met/meff2Jet", ["SS"])
        #         #     iPop = myTopLvl.validationChannels.index("SS_metmeff2Jet")
        #         #     myTopLvl.validationChannels.pop(iPop)
        #         # else:
        #         #     mm2J = myTopLvl.addChannel("met/meff2Jet", ["SS"], 5, 0.2, 0.7)
        #         #     mm2J.useOverflowBin = True
        #         #     mm2J.addSystematic(jes)
        #         #     pass
        #         #myTopLvl.addSignalChannels([mm2J])
        #
