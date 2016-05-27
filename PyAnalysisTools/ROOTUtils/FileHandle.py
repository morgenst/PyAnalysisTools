__author__ = 'marcusmorgenstern'
__mail__ = ''

import os
from ROOT import TFile


class FileHandle(object):
    def __init__(self, fName, sPath = './'):
        self.fName = fName
        self.path = sPath
        self.absFName = os.path.join(self.path, self.fName)
        self.open()

    def open(self):
        if not os.path.exists(self.absFName):
            raise ValueError("File " + os.path.join(self.path, self.fName) + " does not exist.")

        self.tfile = TFile.Open(os.path.join(self.path, self.fName), 'READ')

    def get_objects(self):
        objects = []
        for obj in self.tfile.GetListOfKeys():
            objects.append(self.tfile.Get(obj.GetName()))
        return objects

    def get_objects_by_type(self, typename):
        obj = self.get_objects()
        obj = filter(lambda t: t.InheritsFrom(typename), obj)
        return obj

    def get_object_by_name(self, obj_name):
        obj = self.tfile.Get(obj_name)
        if not obj.__nonzero__():
            raise ValueError("Object " + obj_name + " does not exist in file " + os.path.join(self.path, self.fName))
        self.release_object_from_file(obj)
        return obj

    def release_object_from_file(self, obj):
        obj.SetDirectory(0)
