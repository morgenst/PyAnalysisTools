__author__ = 'marcusmorgenstern'
__mail__ = 'marcus.matthias.morgenstern@cern.ch'


# helper methods to deal with objects


def get_objects_from_canvas(canvas):
    # todo: add logger warning for empty canvas
    obj = [canvas.GetPrimitive(key.GetName()) for key in canvas.GetListOfPrimitives()]
    return obj


def get_objects_from_canvas_by_type(canvas, typename):
    obj = get_objects_from_canvas(canvas)
    obj = filter(lambda o: o.InheritsFrom(typename), obj)
    return obj
