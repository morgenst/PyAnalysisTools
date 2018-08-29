class PlotableObject():
    def __init__(self, plot_object = None, is_ref = True, ref_id = -1, label = "", cuts = None, process=None):
        self.plot_object = plot_object
        self.is_ref = is_ref
        self.ref_id = ref_id
        self.label = label
        self.cuts = cuts
        self.process = process
