__author__ = 'marcusmorgenstern'
__mail__ = ''

import os
from ROOT import TFile

class FileHandle(object):
    def __init__(self, fName, sPath = './'):
        self.fName = fName
        self.path = sPath
        self.absFName = os.path.join(self.path, self.fName)

    def open(self):
        if not os.path.exists(self.absFName):
            raise ValueError("File " + os.path.join(self.path, self.fName) + " does not exist.")

        self.file = TFile.Open(os.path.join(self.path, self.fName), 'READ')

    def getObjects(self):
        objects = []
        for obj in self.file.GetListOfKeys():
            objects.append(self.file.Get(obj.GetName()))
        return objects

    def getObjectsByType(self, typename):
        obj = self.getObjects()
        obj = filter(lambda t: t.InheritsFrom(typename), obj)
        return obj