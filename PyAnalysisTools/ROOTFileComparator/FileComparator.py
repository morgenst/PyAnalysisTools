__author__ = 'marcusmorgenstern'
__mail__ = ''

from ROOT import TFile

class FileComparator(object):
    def __init__(self, args):
        self.lhs = args.lhs
        self.rhs = args.rhs
        self.outdir = args.outdir

    def __del__(self):
        self.lhs.Close()
        self.rhs.Close()

    def parse_files(self):
        self.content = {}
        self.__parseFile(self.lhs)

    def __parseFile(self, file):
        for item in file.GetListOfKeys():
            print item