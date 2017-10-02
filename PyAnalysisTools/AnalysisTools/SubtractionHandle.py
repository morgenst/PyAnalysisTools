from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config


class SubtractionHandle(object):
    def __init__(self, **kwargs):
        self.subtract_items = kwargs["subtract_items"]
        self.output_name = kwargs["output_name"]
        self.reference_item = kwargs["reference_item"]
        kwargs.setdefault("process_configs", None)
        self.process_configs = kwargs["process_configs"]
        self.type = "DataProvider"
        
    def execute(self, histograms):
        try:
            key, value = filter(lambda kv: kv[0].name == self.reference_item, histograms.iteritems())[0]
        except IndexError:
            raise InvalidInputError("Could not find reference item {:s} in ".format(self.reference_item), histograms)
        output = histograms.pop(key)
        for item in self.subtract_items:
            try:
                key = filter(lambda kv: kv.name == item, histograms.keys())[0]
            except IndexError:
                _logger.warn("Could not find item {:s} in histograms".format(item))
            output.Add(histograms.pop(key), -1)
        output_key = self.output_name
        if self.process_configs is not None:
            process_config = find_process_config(self.output_name, self.process_configs)
            if process_config is not None:
                output_key = process_config
        histograms[output_key] = output
        return histograms


