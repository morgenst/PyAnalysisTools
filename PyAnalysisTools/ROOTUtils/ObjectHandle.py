from builtins import map
import re
from PyAnalysisTools.base import _logger


def get_objects_from_canvas(canvas):
    """
    Get all objects draw in canvas
    :param canvas: canvas
    :type canvas: ROOT.TCanvas
    :return: list of objects (primitives) in canvas
    :rtype: list
    """

    if len(canvas.GetListOfPrimitives()) == 0:
        _logger.warning('Provided empty canvas')
        return []
    obj = [canvas.GetPrimitive(key.GetName()) for key in canvas.GetListOfPrimitives()]
    return obj


def get_objects_from_canvas_by_type(canvas, typename):
    """
    Read objects in canvas and filter by typename
    :param canvas: canvas
    :type canvas: TCanvas
    :param typename: type of object to be filtered
    :type typename: str
    :return: objects of type typename in canvas
    :rtype: list[typename]
    """
    obj = get_objects_from_canvas(canvas)
    obj = [o for o in obj if o is not None]
    if isinstance(typename, list):
        obj = [o for o in obj if any(o.InheritsFrom(t) for t in typename)]
    else:
        obj = [o for o in obj if o.InheritsFrom(typename)]
    return obj


def get_objects_from_canvas_by_name(canvas, name):
    """
    Get all objects draw in canvas matching name
    :param canvas: canvas
    :type canvas: ROOT.TCanvas
    :param name: name(s) of objects which want to be retrieved
    :type name: str or list of str
    :return: list of matching objects
    :rtype: list (None)
    """

    objects = get_objects_from_canvas(canvas)
    if not isinstance(name, list):
        name = [name]
    patterns = list(map(re.compile, name))
    objects = [obj for obj in objects if any(re.search(pattern, obj.GetName()) for pattern in patterns)]
    if len(objects) == 0:
        return None
    return objects


def merge_objects_by_process_type(canvas, process_config, merge_type):
    objects = get_objects_from_canvas_by_type(canvas, "TH1")
    if len(objects) == 0:
        return None
    first_object = objects[0]
    variable = "_".join(first_object.GetName().split("_")[0:-1])
    merged_hist = first_object.Clone("_".join([variable, merge_type]))
    for obj in objects[1:]:
        process_name = obj.GetName().split("_")[-1]
        if process_name not in process_config:
            if 'unc' not in obj.GetName():
                _logger.warning('Could not find process {:s} in process_config'.format(process_name))
            continue
        if not process_config[process_name].type == merge_type:
            _logger.debug('Found process {:s} in configs but type {:s} does not '
                          'match requested type {:s}'.format(process_name, process_config[process_name].type,
                                                             merge_type))
            continue
        merged_hist.Add(obj)
    return merged_hist


def find_branches_matching_pattern(tree, pattern):
    """
    Parse branches in TTree and return a list of all branches matching regex defined in pattern
    :param tree: tree
    :type tree: ROOT.TTree
    :param pattern: regular expression used to find matching branch names
    :type pattern: str
    :return: list of matched branch names
    :rtype:
    """

    pattern = re.compile(pattern)
    branch_names = []
    for branch in tree.GetListOfBranches():
        if re.search(pattern, branch.GetName()):
            branch_names.append(branch.GetName())
    if len(branch_names) == 0:
        _logger.warning("Could not find objects matching %s in %s" % (pattern, tree.GetName()))
    return branch_names
