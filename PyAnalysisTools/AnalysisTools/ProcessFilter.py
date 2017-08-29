class ProcessFilter(object):
    def __init__(self, **kwargs):
        self.processes = kwargs["processes"]
        self.type = "Filter"

    def execute(self, histograms):
        filtered_histograms = {}
        for key, hist in histograms.iteritems():
            if isinstance(key, str):
                process = key
            else:
                process = key.name
            if not any(allowed_process in process for allowed_process in self.processes):
                continue
            filtered_histograms[key] = hist
        return filtered_histograms
