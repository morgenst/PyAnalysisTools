import re
from PyAnalysisTools.base import _logger


def get_objects_from_canvas(canvas):
    # todo: add logger warning for empty canvas
    obj = [canvas.GetPrimitive(key.GetName()) for key in canvas.GetListOfPrimitives()]
    return obj


def get_objects_from_canvas_by_type(canvas, typename):
    obj = get_objects_from_canvas(canvas)
    obj = filter(lambda o: o.InheritsFrom(typename), obj)
    return obj


def get_objects_from_canvas_by_name(canvas, name):
    objects = get_objects_from_canvas(canvas)
    if not isinstance(name, list):
        name = [name]
    patterns = map(re.compile, name)
    objects = filter(lambda obj: any(re.search(pattern, obj.GetName()) for pattern in patterns), objects)
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
    for obj in objects:
        process_name = obj.GetName().split("_")[-1]
        if not process_config[process_name].type == merge_type:
            continue
        merged_hist.Add(obj)
    return merged_hist


def find_branches_matching_pattern(tree, pattern):
    pattern = re.compile(pattern)
    branch_names = []
    for branch in tree.GetListOfBranches():
        if re.search(pattern, branch.GetName()):
            branch_names.append(branch.GetName())
    if len(branch_names) == 0:
        _logger.warning("Could not find objects matching %s in %s" % (pattern, tree.GetName()))
    return branch_names
