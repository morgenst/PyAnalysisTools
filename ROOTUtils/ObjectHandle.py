__author__ = 'marcusmorgenstern'
__mail__ = 'marcus.matthias.morgenstern@cern.ch'

# helper methods to deal with objects


def getObjectsFromCanvas(canvas):
    #todo: add logger warning for empty canvas
    obj = [canvas.GetPrimitive(key.GetName()) for key in canvas.GetListOfPrimitives()]
    return obj

def getObjectsFromCanvasByType(canvas, typename):
    obj = getObjectsFromCanvas(canvas)
    obj = filter(lambda o: o.InheritsFrom(typename), obj)
    return obj