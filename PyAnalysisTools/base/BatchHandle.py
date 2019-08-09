class BatchHandle(object):
    """
    Class to handle submission and monitoring to batch system
    """
    def __init__(self, **kwargs):
        self.system = 'qsub'
        self.queue = 'short7'

    def register_master(self):
        pass

    def submit_childs(self):
        pass