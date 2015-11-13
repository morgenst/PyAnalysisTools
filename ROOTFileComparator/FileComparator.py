__author__ = 'marcusmorgenstern'
__mail__ = ''

class FileComparator(object):
    def __init__(self, args):
        self.lhs = args.lhs
        self.rhs = args.rhs
        self.outdir = args.outdir

    def parseFiles(self):


        """