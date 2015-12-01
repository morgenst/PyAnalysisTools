__author__ = 'marcusmorgenstern'
__mail__ = ''

from ROOT import TFile

class FileComparator(object):
    def __init__(self, args):
        self.lhs = TFile.Open(args.lhs, 'READ')
        self.rhs = TFile.Open(args.rhs, 'READ')
        self.outdir = args.outdir

    def __del__(self):
        self.lhs.Close()
        self.rhs.Close()

    def parseFiles(self):
        self.content = {}
        self.__parseFile(self.lhs)

    def __parseFile(self, file):
        for item in file.GetListOfKeys():
            print item