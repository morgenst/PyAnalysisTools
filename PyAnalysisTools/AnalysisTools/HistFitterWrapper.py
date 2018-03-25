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
        self.fit_type = "excl"
        self.my_fit_type = None

        kwargs.setdefault("interactive", False)
        kwargs.setdefault("fit", True)
        kwargs.setdefault("fitname", "")
        kwargs.setdefault("minosPars", "")
        kwargs.setdefault("limit_plot", False)
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
        FitType = configMgr.FitType  # enum('FitType','Discovery , Exclusion , Background')
        #myFitType = FitType.Background

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                setattr(self, key, val)
        self.limit_config = LimiConfig(kwargs["limit_config_file"])
        self.process_configs = parse_and_build_process_config(self.process_config_file)


        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=os.path.abspath(self.xs_config_file)) for fn in self.input_files]
        configMgr.histCacheFile = os.path.join(self.output_dir, "data/" + configMgr.analysisName + ".root")
        configMgr.outputFileName = os.path.join(self.output_dir, "results/" + configMgr.analysisName + "_Output.root")
        self.prepare_output()

        self.expand_process_configs()
        self.samples = {}

    def initialise(self):
        if self.fit_type == "bkg":
            self.my_fit_type = self.fit_type.Background
            _logger.info("Will run in background-only fit mode")

        elif self.fit_type == "excl" or self.fit_type == "model-dep":
            self.my_fit_type = configMgr.FitType.Exclusion
            _logger.info("Will run in exclusion (model-dependent) fit mode")
        elif self.fit_type == "disc" or self.fit_type == "model-indep":
            self.my_fit_type = self.fit_type.Discovery
            _logger.info("Will run in discovery (model-independent) fit mode")
        configMgr.myFitType = self.my_fit_type
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
        if self.limit_plot:
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

        if self.run_toys and configMgr.nTOYs > 0 and self.hypotest == False and self.limit_plot == False and self.fit == False:
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

        self.name = kwargs["name"]
        self.output_dir = kwargs["output_dir"]
        self.bkg_name = kwargs["bkg_name"]
        ndata = 7.  # Number of events observed in data
        nbkg = 5.  # Number of predicted bkg events
        nsig = 5.  # Number of predicted signal events
        nbkg_err = 1.  # (Absolute) Statistical error on bkg estimate
        nsig_err = 2.  # (Absolute) Statistical error on signal estimate
        lumi_error = 0.039  # Relative luminosity uncertainty

        configMgr.cutsDict["UserRegion"] = 1.
        configMgr.weights = "1." #todo: that looks weird
        # Set uncorrelated systematics for bkg and signal (1 +- relative uncertainties)
        ucb = Systematic("ucb", configMgr.weights, 1.2, 0.8, "user", "userOverallSys")
        ucs = Systematic("ucs", configMgr.weights, 1.1, 0.9, "user", "userOverallSys")

        # correlated systematic between background and signal (1 +- relative uncertainties)
        corb = Systematic("cor", configMgr.weights, [1.1], [0.9], "user", "userHistoSys")
        cors = Systematic("cor", configMgr.weights, [1.15], [0.85], "user", "userHistoSys")

        # Setting the parameters of the hypothesis test
        configMgr.doExclusion = True  # True=exclusion, False=discovery
        # configMgr.nTOYs=5000
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
        bkgSample.addSystematic(corb)
        bkgSample.addSystematic(ucb)

        sigSample = Sample("Sig", kPink)
        sigSample.setNormFactor("mu_Sig", 1., 0., 100.)
        sigSample.setStatConfig(True)
        sigSample.setNormByTheory()
        sigSample.buildHisto([nsig], "UserRegion", "cuts", 0.5)
        sigSample.buildStatErrors([nsig_err], "UserRegion", "cuts")
        sigSample.addSystematic(cors)
        sigSample.addSystematic(ucs)

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

        # These lines are needed for the user analysis to run
        # Make sure file is re-made when executing HistFactory
        if configMgr.executeHistFactory:
            file_name = os.path.join(self.output_dir, "data", "{s}.root".format(configMgr.analysisName))
            if os.path.isfile(file_name):
                os.remove(file_name)



class HistFitterShapeAnalysis(HistFitterWrapper):

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ShapeAnalysis")
        kwargs.setdefault("read_tree", True)
        kwargs.setdefault("create_workspace", True)
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
        super(HistFitterShapeAnalysis, self).__init__(**kwargs)

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

class HistFitterShapeAnalysisOld(HistFitterWrapper):

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ShapeAnalysis")
        kwargs.setdefault("read_tree", True)
        kwargs.setdefault("create_workspace", True)
        kwargs.setdefault("output_dir", kwargs["output_dir"])
        configMgr.calculatorType = 2
        configMgr.testStatType = 3
        configMgr.nPoints = 20
        FitType = configMgr.FitType
        configMgr.writeXML = True
        self.analysis_name = kwargs["name"]
        self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
        # ------------------------------------------------------------------------------------------------------
        # Possibility to blind the control, validation and signal regions.
        # We only have one signal region in this config file, thus only blinding the signal region makes sense.
        # the other two commands are only given for information here.
        # ------------------------------------------------------------------------------------------------------

        configMgr.blindSR = False  # Blind the SRs (default is False)
        configMgr.blindCR = False  # Blind the CRs (default is False)
        configMgr.blindVR = False  # Blind the VRs (default is False)
        # configMgr.useSignalInBlindedData = True
        make_dirs(self.output_dir)
        cur_dir = os.path.abspath(os.path.curdir)
        os.chdir(self.output_dir)
        # ---------------------------------------------------
        # Specify the default signal point
        # Others to be given via option -g via command line
        # ---------------------------------------------------

        if not 'sigSamples' in dir():
            sigSamples = ["LQ"]

        # -------------------------------------
        # Now we start to build the data model
        # -------------------------------------

        # First define HistFactory attributes
        configMgr.analysisName = self.analysis_name
        make_dirs(os.path.join(self.output_dir, "results"))
        make_dirs(os.path.join(self.output_dir, "data"))
        make_dirs(os.path.join(self.output_dir, "config"))
        copy(os.path.join(os.environ["HISTFITTER"], "config/HistFactorySchema.dtd"),
                          os.path.join(self.output_dir, "config/HistFactorySchema.dtd"))
        configMgr.histCacheFile = os.path.join(self.output_dir, "data/" + configMgr.analysisName + ".root")
        configMgr.outputFileName = os.path.join(self.output_dir, "results/" + configMgr.analysisName + "_Output.root")
        # activate using of background histogram cache file to speed up processes
        # configMgr.useCacheToTreeFallback = True # enable the fallback to trees
        # configMgr.useHistBackupCacheFile = True # enable the use of an alternate data file
        # configMgr.histBackupCacheFile =  "data/MyShapeFitExample_template.root" # the data file of your previous fit (= the backup cache file)


        # Scaling calculated by outputLumi / inputLumi
        configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        configMgr.setLumiUnits("fb-1")

        # Set the files to read from
        # bgdFiles = []
        # if kwargs["read_tree"]:
        #     bgdFiles.append("/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364165_0.root")
        #     #bgdFiles.append("/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_OneSoftEle_BG_v3.root")
        #     #bgdFiles.append("/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_OneSoftMuo_BG_v3.root")
        # else:
        #     bgdFiles = [configMgr.histCacheFile]
        #     pass
        # configMgr.setFileList(bgdFiles)

        # Dictionnary of cuts for Tree->hist
        #configMgr.cutsDict["SR"] = "((lep1Pt < 20 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7) || (lep1Pt < 25 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6))"
        configMgr.cutsDict["SR"] = "(electron_pt > 65000.)"
        # Tuples of nominal weights without and with b-jet selection
        #configMgr.weights = ["genWeight", "eventWeight", "leptonWeight", "triggerWeight", "truthWptWeight","bTagWeight2Jet"]
        configMgr.weights = ["weight"]

        # QCD weights without and with b-jet selection
        # we turn the QCD background of for the tutorial as we do not want to use ATLAS data
        # configMgr.weightsQCD = "qcdWeight"
        # configMgr.weightsQCDWithB = "qcdBWeight"

        # --------------------
        # List of systematics
        # --------------------

        # Alpgen KtScale (weight-based)
        # ktScaleWHighWeights = ("genWeight", "eventWeight", "ktfacUpWeightW", "bTagWeight2Jet")
        # ktScaleWLowWeights = ("genWeight", "eventWeight", "ktfacDownWeightW", "bTagWeight2Jet")
        # wzKtScale = Systematic("KtScaleWZ", configMgr.weights, ktScaleWHighWeights, ktScaleWLowWeights, "weight",
        #                        "overallSys")
        #
        # ktScaleTopHighWeights = ("genWeight", "eventWeight", "ktfacUpWeightTop", "bTagWeight2Jet")
        # ktScaleTopLowWeights = ("genWeight", "eventWeight", "ktfacDownWeightTop", "bTagWeight2Jet")
        # # topKtScale = Systematic("KtScaleTop",configMgr.weights,ktScaleTopHighWeights,ktScaleTopLowWeights,"weight","overallSys")
        # topKtScale = Systematic("KtScaleTop", configMgr.weights, ktScaleTopHighWeights, ktScaleTopLowWeights, "weight",
        #                         "histoSys")
        # topKtScale = Systematic("KtScaleTop",configMgr.weights,ktScaleTopHighWeights,ktScaleTopLowWeights,"weight","normHistoSys")

        # JES (tree-based)
        # jes = Systematic("JES", "_NoSys", "_JESup", "_JESdown", "tree", "overallSys")
        # configMgr.nomName = "_NoSys"

        # -------------------------------------------
        # List of samples and their plotting colours
        # -------------------------------------------
        # wjetsSample = Sample("wjets", kGreen - 9)
        # wjetsSample.setFile("/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364165_0.root")
        # wjetsSample.setTreeName("Nominal/BaseSelection_lq_tree_Final")
        # wjetsSample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)
        # # topSample.setNormFactor("mu_Top",1.,0.,5.)
        # #wzSample = Sample("WZ", kAzure + 1)
        # # wzSample.setNormFactor("mu_WZ",1.,0.,5.)
        # dataSample = Sample("Data", kBlack)
        # dataSample.setData()
        # dataSample.setFile("/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-data16_13TeV_periodB_0.root")
        # dataSample.setTreeName("Nominal/BaseSelection_lq_tree_Final")
        # dataSample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)
        # #dataSample.buildStatErrors([1.,1.,2.4,3.9,2.,0.],"SR","metmeff2Jet")

        # **************
        # Exclusion fit
        # **************
        if True: #myFitType == FitType.Exclusion:

            # loop over all signal points
            for sig in sigSamples:
                # Fit config instance
                exclusionFitConfig = configMgr.addFitConfig("Exclusion_" + sig)
                meas = exclusionFitConfig.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=0.039)
                meas.addPOI("mu_SIG")

                # Samples
                #exclusionFitConfig.addSamples([wjetsSample, dataSample])
                exclusionFitConfig.addSamples(self.samples.values())
                # Systematics
                #exclusionFitConfig.getSample("Top").addSystematic(topKtScale)
                #exclusionFitConfig.getSample("WZ").addSystematic(wzKtScale)
                #exclusionFitConfig.addSystematic(jes)

                #Channel
                srBin = exclusionFitConfig.addChannel("lq_mass_max", ["SR"], 6, 1000., 2000.)
                srBin.useOverflowBin = True
                srBin.useUnderflowBin = True
                exclusionFitConfig.addSignalChannels([srBin])

                sigSample = Sample(sig, kPink)
                #sigSample.setFileList(["/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_p832_GG-One-Step_soft_v1.root"])
                sigSample.setFileList(["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"])
                sigSample.buildHisto([0., 1., 5., 15., 4., 0.], "SR", "lq_mass_max", 0.1, 0.1)
                sigSample.setNormByTheory()
                sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
                # sigSample.addSampleSpecificWeight("0.001")
                exclusionFitConfig.addSamples(sigSample)
                exclusionFitConfig.setSignalSample(sigSample)

                # Cosmetics
                # srBin.minY = 0.0001
                # srBin.maxY = 80.

        super(HistFitterShapeAnalysis, self).__init__(**kwargs)
        self.input_files = kwargs["input_files"]

        self.FitType = configMgr.FitType
        # First define HistFactory attributes
        # Scaling calculated by outputLumi / inputLumi
        configMgr.inputLumi = 0.001  # Luminosity of input TTree after weighting
        configMgr.outputLumi = 4.713  # Luminosity required for output histograms
        configMgr.setLumiUnits("fb-1")
        configMgr.calculatorType = 2
        #configMgr.histCacheFile = "data/" + configMgr.analysisName + ".root"


        # Set the files to read from
        # bgdFiles = []
        # sigFiles = []
        # if configMgr.readFromTree:
        #     bgdFiles.append("/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364165_0.root")
        #     #bgdFiles.append("/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_OneSoftEle_BG_v3.root")#self.input_files[0])
        #     if self.my_fit_type == configMgr.FitType.Exclusion:
        #         # 1-step simplified model
        #         sigFiles.append("/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root")
        useStat = True
        # Dictionnary of cuts for Tree->hist
        # CR
        # configMgr.cutsDict[
        #     "SLWR"] = "(lep1Pt < 20 && lep2Pt<10 && met>180 && met<250 && mt>40 && mt<80 && nB2Jet==0 && jet1Pt>130 && jet2Pt>25  && AnalysisType==7) || (lep1Pt < 25 && lep2Pt<10 && met>180 && met<250 && mt>40 && mt<80 && nB2Jet==0 && jet1Pt>130 && jet2Pt>25  && AnalysisType==6)"
        # configMgr.cutsDict[
        #     "SLTR"] = "(lep1Pt < 25 && lep2Pt<10 && met>180 && met<250 && mt>40 && mt<80 && nB2Jet>0 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6) || (lep1Pt < 20 && lep2Pt<10 && met>180 && met<250 && mt>40 && mt<80 && nB2Jet>0 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7)"
        # # VR
        # configMgr.cutsDict[
        #     "SLVR2"] = "(lep1Pt < 25 && lep2Pt<10 && met>180 && met<250 && mt>80 && mt<100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6) || (lep1Pt < 20 && lep2Pt<10 && met>180 && met<250 && mt>80 && mt<100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7)"
        # # SR
        # configMgr.cutsDict[
        #     "SS"] = "((lep1Pt < 20 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7) || (lep1Pt < 25 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6))"
        # configMgr.cutsDict[
        #     "SSloose"] = "((lep1Pt < 20 && lep2Pt<10 && met>200 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7) || (lep1Pt < 25 && lep2Pt<10 && met>200 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6))"
        # configMgr.cutsDict["SR1sl2j"] = configMgr.cutsDict["SS"] + "&& met/meff2Jet>0.3"

        # Tuples of nominal weights without and with b-jet selection
        configMgr.weights = ("weight")

        # name of nominal histogram for systematics
        configMgr.nomName = "_NoSys"

        # List of samples and their plotting colours
        # topSample = Sample("Top", kGreen - 9)
        # topSample.buildHisto([86., 66., 62., 35., 11., 7., 2., 0.], "SLTR", "nJet", 2)
        #topSample.setFileList(["/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_p832_GG-One-Step_soft_v1.root"])
        #topSample.setNormFactor("mu_Top", 1., 0., 5.)
        #topSample.setStatConfig(useStat)
        #topSample.setNormRegions([("SLWR", "nJet"), ("SLTR", "nJet")])
        # wzSample = Sample("WZ", kAzure + 1)
        # wzSample.setNormFactor("mu_WZ", 1., 0., 5.)
        # wzSample.setStatConfig(useStat)
        # wzSample.setNormRegions([("SLWR", "nJet"), ("SLTR", "nJet")])
        # bgSample = Sample("BG", kYellow - 3)
        # bgSample.setNormFactor("mu_BG", 1., 0., 5.)
        # bgSample.setStatConfig(useStat)
        # bgSample.setNormRegions([("SLWR", "nJet"), ("SLTR", "nJet")])
        # #
        # qcdSample = Sample("QCD", kGray + 1)
        # qcdSample.setQCD(True, "histoSys")
        # qcdSample.setStatConfig(useStat)
        # #
        # dataSample = Sample("Data", kBlack)
        # dataSample.setData()
        # dataSample.buildHisto([86., 66., 62., 35., 11., 7., 2., 0.], "SLTR", "nJet", 2)
        # #dataSample.buildHisto([1092., 426., 170., 65., 27., 9., 4., 1.], "SLWR", "nJet", 2)
        #
        # # set the file from which the samples should be taken
        # for sam in [topSample, dataSample]:
        #     sam.setFileList(bgdFiles)

        # Binnings
        nJetBinLowHard = 3
        # nJetBinLowSoft = 2
        # nJetBinHighTR = 10
        # nJetBinHighWR = 10

        nBJetBinLow = 0
        nBJetBinHigh = 4

        meffNBins = 6
        meffBinLow = 400.
        meffBinHigh = 1600.

        meffNBinsSR4 = 4
        meffBinLowSR4 = 800.
        meffBinHighSR4 = 1600.

        lepPtNBins = 6
        lepPtLow = 20.
        lepPtHigh = 600.

        srNBins = 1
        srBinLow = 0.5
        srBinHigh = 1.5

        # ************
        # Bkg only fit
        # ************

        # bkt = configMgr.addFitConfig("BkgOnly")
        # if useStat:
        #     bkt.statErrThreshold = 0.05
        # else:
        #     bkt.statErrThreshold = None
        # bkt.addSamples([topSample, dataSample])
        # #configMgr.setFileList(["/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_OneSoftEle_BG_v3.root",
        # #                  "/afs/cern.ch/user/m/morgens/afs_work/devarea/HF2/HistFitter/samples/tutorial/SusyFitterTree_OneSoftMuo_BG_v3.root"])
        # # configMgr.cutsDict[
        # #     "SR"] = "((lep1Pt < 20 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==7) || (lep1Pt < 25 && lep2Pt<10 && met>250 && mt>100 && jet1Pt>130 && jet2Pt>25 && AnalysisType==6))"
        #
        # # Systematics to be applied globally within this topLevel
        # # bkt.getSample("Top").addSystematic(topKtScale)
        # # bkt.getSample("WZ").addSystematic(wzKtScale)
        #
        # meas = bkt.addMeasurement(name="NormalMeasurement", lumi=1.0, lumiErr=0.039)
        # meas.addPOI("mu_SIG")
        # #meas.addParamSetting("mu_BG", True, 1)
        # #meas.addParamSetting("Lumi", True, 1)
        # srBin = bkt.addChannel("SR", ["SR"], 6, 0.1, 0.7)
        # bkt.addSignalChannels([srBin])

        # -------------------------------------------------
        # Constraining regions - statistically independent
        # -------------------------------------------------

        # WR using nJet
        # nJetWS = bkt.addChannel("nJet", ["SLWR"], nJetBinHighWR - nJetBinLowSoft, nJetBinLowSoft, nJetBinHighWR)
        # nJetWS.hasB = True
        # nJetWS.hasBQCD = False
        # nJetWS.useOverflowBin = False
        # nJetWS.addSystematic(jes)

        # TR using nJet
        # nJetTS = bkt.addChannel("nJet", ["SLTR"], nJetBinHighTR - nJetBinLowSoft, nJetBinLowSoft, nJetBinHighTR)
        # nJetTS.hasB = True
        # nJetTS.hasBQCD = True
        # nJetTS.useOverflowBin = False
        # nJetTS.addSystematic(jes)

        #bkt.addBkgConstrainChannels([nJetWS, nJetTS])

        ### alternative: statistical error for each sample
        # nJetWS.getSample("Top").addSystematic(statWRtop)
        # nJetWS.getSample("WZ").addSystematic(statWRwz)

        ###################
        #                                               #
        #    Example new cosmetics     #
        #                                               #
        ###################

        # Set global plotting colors/styles
        # bkt.dataColor = dataSample.color
        # bkt.totalPdfColor = kBlue
        # bkt.errorFillColor = kBlue - 5
        # bkt.errorFillStyle = 3004
        # #bkt.errorLineStyle = kDashed
        # bkt.errorLineColor = kBlue - 5

        # Set Channel titleX, titleY, minY, maxY, logY
        # nJetWS.minY = 0.5
        # nJetWS.maxY = 5000
        # nJetWS.titleX = "n jets"
        # nJetWS.titleY = "Entries"
        # nJetWS.logY = True
        # nJetWS.ATLASLabelX = 0.25
        # nJetWS.ATLASLabelY = 0.85
        # nJetWS.ATLASLabelText = "Work in progress"

        # --------------------------------------------------------------
        # Validation regions - not necessarily statistically independent
        # --------------------------------------------------------------
        if self.validation:
            # s1l2jT
            srs1l2jTChannel = bkt.addChannel("cuts", ["SR1sl2j"], srNBins, srBinLow, srBinHigh)
            srs1l2jTChannel.addSystematic(jes)

            # additional VRs if using soft lep CRs
            nJetSLVR2 = bkt.addChannel("nJet", ["SLVR2"], nJetBinHighTR - nJetBinLowSoft, nJetBinLowSoft, nJetBinHighTR)
            nJetSLVR2.addSystematic(jes)

            # signal region treated as validation region for this case
            mm2J = bkt.addChannel("met/meff2Jet", ["SS"], 6, 0.1, 0.7)
            mm2J.useOverflowBin = True
            mm2J.addSystematic(jes)
            mm2J.remapSystChanName = 'metmeff2Jet_SSloose'

            # signal region treated as validation region for this case
            mm2Jl = bkt.addChannel("met/meff2Jet", ["SSloose"], 6, 0.1, 0.7)
            mm2Jl.useOverflowBin = True
            mm2Jl.addSystematic(jes)

            #    bkt.addValidationChannels([nJetSLVR2,metSLVR2,meffSLVR2,nBJetSLVR2,metmeffSLVR2,mm2J,srs1l2jTChannel])
            bkt.addValidationChannels([nJetSLVR2, srs1l2jTChannel, mm2J, mm2Jl])

            dataSample.buildHisto([0., 1., 6., 16., 3., 0.], "SS", "metmeff2Jet", 0.1, 0.1)
            dataSample.buildHisto([25.], "SR1sl2j", "cuts", 0.5)
            dataSample.buildHisto([1., 6., 24., 37., 7., 0.], "SSloose", "metmeff2Jet", 0.1, 0.1)
            dataSample.buildHisto([403., 202., 93., 39., 11., 10., 4., 1.], "SLVR2", "nJet", 2)

        # **************
        # Discovery fit
        # **************

        # if myFitType == FitType.Discovery:
        #     discovery = configMgr.addFitConfigClone(bkt, "Discovery")
        #
        #     # s1l2jT = signal region/channel
        #     ssChannel = discovery.addChannel("cuts", ["SS"], srNBins, srBinLow, srBinHigh)
        #     ssChannel.addSystematic(jes)
        #     ssChannel.addDiscoverySamples(["SS"], [1.], [0.], [100.], [kMagenta])
        #     discovery.addSignalChannels([ssChannel])
        #     dataSample.buildHisto([26.], "SS", "cuts", 0.5)

        # -----------------------------
        # Exclusion fits (1-step simplified model in this case)
        # -----------------------------
        doValidation = False
        if True: #myFitType == FitType.Exclusion:
            sigSamples = ["/eos/atlas/user/m/morgens/datasets/LQ/ntuples/v2/ntuple-364131_0.root"]
            dataSample.buildHisto([1., 6., 16., 3., 0.], "SS", "lq_mass_max", 0.2, 0.1)

            for sig in sigSamples:
                #myTopLvl = configMgr.addFitConfigClone(bkt, "Sig_%s" % sig)
                sigSample = Sample(sig, kPink)
                sigSample.setFileList([sig])
                sigSample.setNormByTheory()
                sigSample.setStatConfig(useStat)
                sigSample.setNormFactor("mu_SIG", 1., 0., 5.)
                # bkt.addSamples(sigSample)
                # bkt.setSignalSample(sigSample)

                # s1l2j using met/meff
                # if doValidation:
                #     mm2J = myTopLvl.getChannel("met/meff2Jet", ["SS"])
                #     iPop = myTopLvl.validationChannels.index("SS_metmeff2Jet")
                #     myTopLvl.validationChannels.pop(iPop)
                # else:
                #     mm2J = myTopLvl.addChannel("met/meff2Jet", ["SS"], 5, 0.2, 0.7)
                #     mm2J.useOverflowBin = True
                #     mm2J.addSystematic(jes)
                #     pass
                #myTopLvl.addSignalChannels([mm2J])
        # Create TLegend (AK: TCanvas is needed for that, but it gets deleted afterwards)
        c = TCanvas()
        compFillStyle = 1001  # see ROOT for Fill styles
        leg = TLegend(0.6, 0.475, 0.9, 0.925, "")
        leg.SetFillStyle(0)
        leg.SetFillColor(0)
        leg.SetBorderSize(0)
        #
        entry = TLegendEntry()
        entry = leg.AddEntry("", "Data 2011 (#sqrt{s}=7 TeV)", "p")
        #entry.SetMarkerColor(bkt.dataColor)
        entry.SetMarkerStyle(20)
        #
        entry = leg.AddEntry("", "Total pdf", "lf")
        #entry.SetLineColor(bkt.totalPdfColor)
        entry.SetLineWidth(2)
        # entry.SetFillColor(bkt.errorFillColor)
        # entry.SetFillStyle(bkt.errorFillStyle)
        #
        # entry = leg.AddEntry("", "t#bar{t}", "lf")
        # entry.SetLineColor(topSample.color)
        # entry.SetFillColor(topSample.color)
        # entry.SetFillStyle(compFillStyle)
        # #
        # entry = leg.AddEntry("", "WZ", "lf")
        # entry.SetLineColor(wzSample.color)
        # entry.SetFillColor(wzSample.color)
        # entry.SetFillStyle(compFillStyle)
        # #
        # entry = leg.AddEntry("", "multijets (data estimate)", "lf")
        # entry.SetLineColor(qcdSample.color)
        # entry.SetFillColor(qcdSample.color)
        # entry.SetFillStyle(compFillStyle)
        # #
        # entry = leg.AddEntry("", "single top & diboson", "lf")
        # entry.SetLineColor(bgSample.color)
        # entry.SetFillColor(bgSample.color)
        # entry.SetFillStyle(compFillStyle)
        #
        # if myFitType == FitType.Exclusion:
        #     entry = leg.AddEntry("", "signal", "lf")
        #     entry.SetLineColor(kPink)
        #     entry.SetFillColor(kPink)
        #     entry.SetFillStyle(compFillStyle)

        # Set legend for fitConfig
        #bkt.tLegend = leg
        # if myFitType == FitType.Exclusion:
        #     myTopLvl.tLegend = leg
        #c.Close()