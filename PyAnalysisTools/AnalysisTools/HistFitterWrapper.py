from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as soh
try:
    from configManager import configMgr
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
from PyAnalysisTools.base.ShellUtils import make_dirs, copy
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config, find_process_config
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

    def prepare_output(self):
        make_dirs(os.path.join(self.output_dir, "results"))
        make_dirs(os.path.join(self.output_dir, "data"))
        make_dirs(os.path.join(self.output_dir, "config"))
        os.chdir(self.output_dir)
        copy(os.path.join(os.environ["HISTFITTER"], "config/HistFactorySchema.dtd"),
                          os.path.join(self.output_dir, "config/HistFactorySchema.dtd"))

    def __init__(self, **kwargs):
        configMgr.analysisName = kwargs["name"]
        self.fit_type = "disc"
        #self.my_fit_type = None

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
        kwargs.setdefault("read_tree", configMgr.readFromTree)
        kwargs.setdefault("create_workspace", configMgr.executeHistFactory)
        #kwargs.setdefault("use_XML", configMgr.writeXML)
        kwargs.setdefault("use_XML", True)
        kwargs.setdefault("num_toys", configMgr.nTOYs)
        kwargs.setdefault("seed", configMgr.toySeed)
        kwargs.setdefault("use_asimov", configMgr.useAsimovSet)
        kwargs.setdefault("run_toys", False)
        kwargs.setdefault("process_config_file", None)
        self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")

        #FitType = configMgr.FitType  # enum('FitType','Discovery , Exclusion , Background')
        #myFitType = FitType.Background

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                setattr(self, key, val)
        configMgr.histCacheFile = os.path.join(self.output_dir, "data/" + configMgr.analysisName + ".root")
        configMgr.outputFileName = os.path.join(self.output_dir, "results/" + configMgr.analysisName + "_Output.root")
        self.prepare_output()

        self.samples = {}

    def parse_configs(self):
        self.limit_config = None
        if hasattr(self, "limit_config_file"):
            self.limit_config = LimiConfig(kwargs["limit_config_file"])
        if self.process_config_file is not None:
            self.process_configs = parse_and_build_process_config(self.process_config_file)
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=os.path.abspath(self.xs_config_file)) for fn in self.input_files]
        self.expand_process_configs()

    def initialise(self):
        if self.fit_type == "bkg":
            configMgr.myFitType = configMgr.FitType.Background
            _logger.info("Will run in background-only fit mode")
        elif self.fit_type == "excl" or self.fit_type == "model-dep":
            configMgr.myFitType = configMgr.FitType.Exclusion
            _logger.info("Will run in exclusion (model-dependent) fit mode")
        elif self.fit_type == "disc" or self.fit_type == "model-indep":
            configMgr.myFitType = configMgr.FitType.Discovery
            _logger.info("Will run in discovery (model-independent) fit mode")
        else:
            _logger.error("fit type not specified. Giving up...")
            exit(0)
        if self.use_archive_histfile:
            configMgr.useHistBackupCacheFile = True

        if self.read_tree:
            configMgr.readFromTree = True

        if self.create_workspace:
            configMgr.executeHistFactory = True

        if self.use_XML:
            configMgr.writeXML = True

        #configMgr.userArg = self.userArg
        configMgr.nTOYs = self.num_toys

        # if self.log_level:
        #     _logger.setLevel(self.log_level, True)

        if self.hypotest:
            configMgr.doHypoTest = True

        if self.discovery_hypotest:
            configMgr.doDiscoveryHypoTest = True

        if self.draw:
            drawArgs = self.draw.split(",")
            if len(drawArgs) == 1 and (drawArgs[0] == "allPlots" or drawArgs[0] == "all"):
                self.draw_before = True
                self.draw_after = True
                drawCorrelationMatrix = True
                drawSeparateComponents = True
                drawLogLikelihood = True
                drawSystematics = True
            elif len(drawArgs) > 0:
                for drawArg in drawArgs:
                    if drawArg == "before":
                        self.draw_before = True
                    elif drawArg == "after":
                        self.draw_after = True
                    elif drawArg == "corrMatrix":
                        drawCorrelationMatrix = True
                    elif drawArg == "sepComponents":
                        drawSeparateComponents = True
                    elif drawArg == "likelihood":
                        drawLogLikelihood = True
                    elif drawArg == "systematics":
                        drawSystematics = True
                    else:
                        _logger.fatal(
                            "Wrong draw argument: '%s'. Possible draw arguments are 'allPlots' or comma separated 'before after corrMatrix sepComponents likelihood'" % drawArg)

        if self.no_empty:
            configMgr.removeEmptyBins = True

        if self.seed != 0:  # 0 is default because type is int
            configMgr.toySeedSet = True
            configMgr.toySeed = self.seed

        if self.use_asimov:
            configMgr.useAsimovSet = True

        # if self.grid_points and self.grid_points != "":
        #     sigSamples = self.grid_points.split(",")
        #     _logger.info("Grid points specified: %s" % sigSamples)

        # if self.regions and self.regions != "" and self.regions != "all":
        #     pickedSRs = self.regions.split(",")
        # else:
        #     pickedSRs = []  # MB: used by 0-lepton fit
        #
        # if len(pickedSRs) > 0:
        #     _logger.info("Selected signal regions: %s" % pickedSRs)

        if self.run_toys:
            runToys = True

        # if self.background:
        #     bkgArgs = self.background.split(',')
        #     if len(bkgArgs) == 2:
        #         configMgr.SetBkgParName(bkgArgs[0])
        #         configMgr.SetBkgCorrVal(float(bkgArgs[1]))
        #         configMgr.SetBkgChlName("")
        #     elif len(bkgArgs) >= 3 and len(bkgArgs) % 3 == 0:
        #         for iChan in xrange(len(bkgArgs) / 3):
        #             iCx = iChan * 3
        #             configMgr.AddBkgChlName(bkgArgs[iCx])
        #             configMgr.AddBkgParName(bkgArgs[iCx + 1])
        #             configMgr.AddBkgCorrVal(float(bkgArgs[iCx + 2]))
        #             continue

        if self.minos:
            minosArgs = self.minos.split(",")
            for idx, arg in enumerate(minosArgs):
                if arg.lower() == "all":
                    minosArgs[idx] = "all"

            self.minosPars = ",".join(minosArgs)

        # if self.constant:
        #     doFixParameters = True
        #     fixedPars = self.constant
        #
        # if self.cmd:
        #     self.info("Python commands executed: %s" % self.cmd)
        #     exec (self.cmd)  ## python execute

        gROOT.SetBatch(not self.interactive)

        """
        mandatory user-defined configuration file
        """
        #execfile(self.configFile[0])  # [0] since any extra arguments (sys.argv[-1], etc.) are caught here

        """
        standard execution from now on
        """

        configMgr.initialize()
        RooRandom.randomGenerator().SetSeed(configMgr.toySeed)
        ReduceCorrMatrix = configMgr.ReduceCorrMatrix

        """
        runs Trees->histos and/or histos->workspace according to specifications
        """
        if configMgr.readFromTree or configMgr.executeHistFactory:
            if self.run_profiling:
                import cProfile
                cProfile.run('configMgr.executeAll()')
            else:
                configMgr.executeAll()

        """
        shows systematics
        """
        if self.drawSystematics:
            from ROOT import Util
            if not os.path.isdir("./plots"):
                _logger.info("no directory './plots' found - attempting to create one")
                os.mkdir("./plots")
            for fC in configMgr.fitConfigs:
                for chan in fC.channels:
                    for sam in chan.sampleList:
                        if not sam.isData:
                            if self.systematics:
                                Systs = self.systematics
                            else:
                                _logger.info("no systematic has been specified.... all the systematics will be considered")
                                print sam.systDict
                                Systs = ""
                                for i in sam.systDict.keys():
                                    Systs += i
                                    if 'Norm' in sam.systDict[i].method:
                                        Systs += "Norm"
                                    Systs += ","
                                if Systs != "":
                                    Systs = Systs[:-1]
                            if Systs != "":
                                Util.plotUpDown(configMgr.histCacheFile, sam.name, Systs,
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

    def run(self):
        """
        runs fitting and plotting, by calling C++ side functions
        """

        if self.fit or self.draw:
            idx = 0
            if len(configMgr.fitConfigs) == 0:
                _logger.fatal("No fit configurations found!")

            runAll = True
            if self.fitname != "":  # user specified a fit name
                fitFound = False
                for (i, config) in enumerate(configMgr.fitConfigs):
                    if configMgr.fitConfigs[i].name == self.fitname:
                        idx = i
                        fitFound = True
                        runAll = False
                        _logger.info("Found fitConfig with name %s at index %d" % (self.fitname, idx))
                        break

                if not fitFound:
                    _logger.fatal("Unable to find fitConfig with name %s, bailing out" % self.fitname)
            noFit = False
            if not self.fit:
                noFit = True
            for i in xrange(len(configMgr.fitConfigs)):
                if not runAll and i != idx:
                    _logger.debug("Skipping fit config {0}".format(configMgr.fitConfigs[i].name))
                    continue

                _logger.info("Running on fitConfig %s" % configMgr.fitConfigs[i].name)
                _logger.info("Setting noFit = {0}".format(noFit))
                self.GenerateFitAndPlotCPP(configMgr.fitConfigs[i], configMgr.analysisName, self.draw_before,
                                               self.draw_after, self.drawCorrelationMatrix, self.drawSeparateComponents,
                                                self.drawLogLikelihood, self.minos, self.minosPars, self.doFixParameters,
                                               self.fixedPars, configMgr.ReduceCorrMatrix, noFit)
            _logger.debug(
                    " GenerateFitAndPlotCPP(configMgr.fitConfigs[%d], configMgr.analysisName, drawBeforeFit, drawAfterFit, drawCorrelationMatrix, drawSeparateComponents, drawLogLikelihood, runMinos, minosPars, doFixParameters, fixedPars, ReduceCorrMatrix)" % idx)
            _logger.debug(
                    "   where drawBeforeFit, drawAfterFit, drawCorrelationMatrix, drawSeparateComponents, drawLogLikelihood, ReduceCorrMatrix are booleans")
            pass

        """
        calculating and printing upper limits for model-(in)dependent signal fit configurations (aka Exclusion/Discovery fit setup)
        """
        if not self.disable_limit_plot:
            for fc in configMgr.fitConfigs:
                if len(fc.validationChannels) > 0:
                    raise (Exception, "Validation regions should be turned off for setting an upper limit!")
                pass
            configMgr.cppMgr.doUpperLimitAll()
            pass

        """
        run exclusion or discovery hypotest
        """
        if self.hypotest or self.discovery_hypotest:
            for fc in configMgr.fitConfigs:
                if len(fc.validationChannels) > 0 and not (fc.signalSample is None or 'Bkg' in fc.signalSample):
                    raise (Exception, "Validation regions should be turned off for doing hypothesis test!")
                pass

            if self.discovery_hypotest:
                configMgr.cppMgr.doHypoTestAll(os.path.join(self.output_dir, 'results/'), False)

            if self.hypotest:
                configMgr.cppMgr.doHypoTestAll(os.path.join(self.output_dir, 'results/'), True)

            pass

        if self.run_toys and configMgr.nTOYs > 0 and self.hypotest is False and self.disable_limit_plot and self.fit is False:
            configMgr.cppMgr.runToysAll()
            pass

        if self.interactive:
            from code import InteractiveConsole
            from ROOT import Util
            cons = InteractiveConsole(locals())
            cons.interact("Continuing interactive session... press Ctrl+d to exit")
            pass

        _logger.info("Leaving HistFitter... Bye!")


    def GenerateFitAndPlotCPP(self, fc, anaName, drawBeforeFit, drawAfterFit, drawCorrelationMatrix, drawSeparateComponents,
                              drawLogLikelihood, minos, minosPars, doFixParameters, fixedPars, ReduceCorrMatrix, noFit):
        """
        function call to top-level C++ side function Util.GenerateFitAndPlot()

        @param fc FitConfig name connected to fit and plot details
        @param anaName Analysis name defined in config file, mainly used for output file/dir naming
        @param drawBeforeFit Boolean deciding whether before-fit plots are produced
        @param drawAfterFit Boolean deciding whether after-fit plots are produced
        @param drawCorrelationMatrix Boolean deciding whether correlation matrix plot is produced
        @param drawSeparateComponents Boolean deciding whether separate component (=sample) plots are produced
        @param drawLogLikelihood Boolean deciding whether log-likelihood plots are produced
        @param minos Boolean deciding whether asymmetric errors are calculated, eg whether MINOS is run
        @param minosPars When minos is called, defining what parameters need asymmetric error calculation
        @param doFixParameters Boolean deciding if some parameters are fixed to a value given or not
        @param fixedPars String of parameter1:value1,parameter2:value2 giving information on which parameter to fix to which value if dofixParameter == True
        @param ReduceCorrMatrix Boolean deciding whether reduced correlation matrix plot is produced
        @param noFit Don't re-run fit but use after-fit workspace
        """

        from ROOT import Util

        _logger.debug('GenerateFitAndPlotCPP: anaName %s ' % anaName)
        _logger.debug("GenerateFitAndPlotCPP: drawBeforeFit %s " % drawBeforeFit)
        _logger.debug("GenerateFitAndPlotCPP: drawAfterFit %s " % drawAfterFit)
        _logger.debug("GenerateFitAndPlotCPP: drawCorrelationMatrix %s " % drawCorrelationMatrix)
        _logger.debug("GenerateFitAndPlotCPP: drawSeparateComponents %s " % drawSeparateComponents)
        _logger.debug("GenerateFitAndPlotCPP: drawLogLikelihood %s " % drawLogLikelihood)
        _logger.debug("GenerateFitAndPlotCPP: minos %s " % minos)
        _logger.debug("GenerateFitAndPlotCPP: minosPars %s " % minosPars)
        _logger.debug("GenerateFitAndPlotCPP: doFixParameters %s " % doFixParameters)
        _logger.debug("GenerateFitAndPlotCPP: fixedPars %s " % fixedPars)
        _logger.debug("GenerateFitAndPlotCPP: ReduceCorrMatrix %s " % ReduceCorrMatrix)
        _logger.debug("GenerateFitAndPlotCPP: noFit {0}".format(noFit))

        Util.GenerateFitAndPlot(fc.name, anaName, drawBeforeFit, drawAfterFit, drawCorrelationMatrix,
                                drawSeparateComponents, drawLogLikelihood, minos, minosPars, doFixParameters, fixedPars,
                                ReduceCorrMatrix, noFit)


class HistFitterCountingExperiment(HistFitterWrapper):
    def __init__(self, **kwargs):
        super(HistFitterCountingExperiment, self).__init__(**kwargs)
        kwargs.setdefault("bkg_name", "Bkg")
        kwargs.setdefault("name", "foo")
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        super(HistFitterCountingExperiment, self).__init__(**kwargs)

        self.name = kwargs["name"]
        self.output_dir = kwargs["output_dir"]
        self.bkg_name = kwargs["bkg_name"]
        ndata = 1.  # Number of events observed in data
        nbkg = 0.911  # Number of predicted bkg events
        nsig = 1.  # Number of predicted signal events
        nbkg_err = 0.376*nbkg  # (Absolute) Statistical error on bkg estimate
        nsig_err = 0.144  # (Absolute) Statistical error on signal estimate
        lumi_error = 0.039  # Relative luminosity uncertainty

        configMgr.cutsDict["UserRegion"] = 1.
        configMgr.weights = "1."
        # Set uncorrelated systematics for bkg and signal (1 +- relative uncertainties)
        # ucb = Systematic("ucb", configMgr.weights, 1.2, 0.8, "user", "userOverallSys")
        # ucs = Systematic("ucs", configMgr.weights, 1.1, 0.9, "user", "userOverallSys")

        # correlated systematic between background and signal (1 +- relative uncertainties)
        # corb = Systematic("cor", configMgr.weights, [1.1], [0.9], "user", "userHistoSys")
        # cors = Systematic("cor", configMgr.weights, [1.15], [0.85], "user", "userHistoSys")

        # Setting the parameters of the hypothesis test
        configMgr.doExclusion = False  # True=exclusion, False=discovery
        configMgr.nTOYs=5000
        configMgr.calculatorType = 2  # 2=asymptotic calculator, 0=frequentist calculator
        configMgr.testStatType = 3  # 3=one-sided profile likelihood test statistic (LHC default)
        configMgr.nPoints = 20  # number of values scanned of signal-strength for upper-limit determination of signal strength.

        configMgr.writeXML = True

        ##########################

        # Give the analysis a name
        configMgr.analysisName = self.name
        configMgr.outputFileName = os.path.join(self.output_dir, "results",
                                                "{:s}_Output.root".format(configMgr.analysisName))
        # Define samples
        bkgSample = Sample(self.bkg_name, kGreen - 9)
        bkgSample.setStatConfig(True)
        bkgSample.buildHisto([nbkg], "UserRegion", "cuts", 0.5)
        bkgSample.buildStatErrors([nbkg_err], "UserRegion", "cuts")
        # bkgSample.addSystematic(corb)
        # bkgSample.addSystematic(ucb)

        sigSample = Sample("Sig", kPink)
        sigSample.setNormFactor("mu_Sig", 1., 0., 100.)
        sigSample.setStatConfig(True)
        sigSample.setNormByTheory()
        sigSample.buildHisto([nsig], "UserRegion", "cuts", 0.5)
        sigSample.buildStatErrors([nsig_err], "UserRegion", "cuts")

        dataSample = Sample("Data", kBlack)
        dataSample.setData()
        dataSample.buildHisto([ndata], "UserRegion", "cuts", 0.5)

        # Define top-level
        ana = configMgr.addFitConfig("SPlusB")
        ana.addSamples([bkgSample, sigSample, dataSample])
        ana.setSignalSample(sigSample)

        # Define measurement
        meas = ana.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=lumi_error)
        meas.addPOI("mu_Sig")
        # meas.addParamSetting("Lumi",True,1)

        # Add the channel
        chan = ana.addChannel("cuts", ["UserRegion"], 1, 0.5, 1.5)
        ana.addSignalChannels([chan])
        self.initialise()

        # These lines are needed for the user analysis to run
        # Make sure file is re-made when executing HistFactory
        if configMgr.executeHistFactory:
            file_name = os.path.join(self.output_dir, "data", "{:s}.root".format(configMgr.analysisName))
            if os.path.isfile(file_name):
                os.remove(file_name)


class HistFitterShapeAnalysis(HistFitterWrapper):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ShapeAnalysis")
        kwargs.setdefault("read_tree", True)
        kwargs.setdefault("create_workspace", True)
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        super(HistFitterShapeAnalysis, self).__init__(**kwargs)
        self.parse_configs()
        configMgr.calculatorType = 2
        configMgr.testStatType = 3
        configMgr.nPoints = 20
        FitType = configMgr.FitType
        configMgr.writeXML = True
        self.analysis_name = kwargs["name"]
        # ------------------------------------------------------------------------------------------------------
        # Possibility to blind the control, validation and signal regions.
        # We only have one signal region in this config file, thus only blinding the signal region makes sense.
        # the other two commands are only given for information here.
        # ------------------------------------------------------------------------------------------------------

        configMgr.blindSR = False  # Blind the SRs (default is False)
        configMgr.blindCR = False  # Blind the CRs (default is False)
        configMgr.blindVR = False  # Blind the VRs (default is False)
        # configMgr.useSignalInBlindedData = True
        cur_dir = os.path.abspath(os.path.curdir)

        # First define HistFactory attributes
        configMgr.analysisName = self.analysis_name

        # Scaling calculated by outputLumi / inputLumi
        configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        configMgr.setLumiUnits("fb-1")

        for channel in self.limit_config.channels:
            configMgr.cutsDict[channel.name] = channel.cuts
        configMgr.cutsDict["SR"] = "(electron_pt > 65000.)"
        configMgr.weights = ["weight"]

        self.build_samples()

        # **************
        # Exclusion fit
        # **************
        if True: #myFitType == FitType.Exclusion:

            # loop over all signal points
            # Fit config instance
            exclusionFitConfig = configMgr.addFitConfig("Exclusion_LQ")
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
        self.FitType = configMgr.FitType
        # First define HistFactory attributes
        # Scaling calculated by outputLumi / inputLumi
        configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        configMgr.setLumiUnits("fb-1")
        configMgr.calculatorType = 2
        #configMgr.histCacheFile = "data/" + configMgr.analysisName + ".root"

        useStat = True
        # Tuples of nominal weights without and with b-jet selection
        configMgr.weights = ("weight")

        # name of nominal histogram for systematics
        configMgr.nomName = "_NoSys"





        # -----------------------------
        # Exclusion fits (1-step simplified model in this case)
        # -----------------------------
        doValidation = False
        # if True: #myFitType == FitType.Exclusion:
        #     sigSamples = ["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"]
        #     #self.dataSample.buildHisto([1., 6., 16., 3., 0.], "SS", "lq_mass_max", 0.2, 0.1)
        #
        #     for sig in sigSamples:
        #         #myTopLvl = configMgr.addFitConfigClone(bkt, "Sig_%s" % sig)
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
