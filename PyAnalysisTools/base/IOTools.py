import ROOT
import os


class Writer:
    def __init__(self, directory=None):
        """

        :param directory:
        """
        if directory is None:
            directory = os.path.abspath(os.curdir)
        self.dir = directory
        self.__check_and_create_directory(self.dir)

    def __check_and_create_directory(self, directory):
        _logger.debug("Check if directory: %s exists" % (directory))
        if not os.path.exists(directory):
            _logger.debug("Create directory: %s exists" % (directory))
            os.makedirs(directory)

    def dump_canvas(self, canvas, message=None, image=None):
        if image:
            self.write_canvas_to_file(canvas, image)
        else:
            if message is None:
                image = raw_input("save canvas as (<RET> for skipping): ")
            else:
                image = raw_input(message)

            if image:
                self.write_canvas_to_file(canvas, image)

    def write_canvas_to_file(self, canvas, name, extension='pdf'):
        ext = self.parse_extension_from_file_name(name)
        if ext is not None:
            extension = ext
            name = ''.join(name.split('.')[0:-1])
        if not extension.startswith('.'):
            extension = '.' + extension
        if extension == '.root':
            self.write_object_to_root_file(canvas, name + extension)
        else:
            canvas.SaveAs(os.path.join(os.path.join(self.dir,
                                                    name + extension)))

    def write_object_to_root_file(self, obj, filename, dir=''):
        f = ROOT.gROOT.GetListOfFiles().FindObject(filename)
        if not f:
            f = ROOT.TFile.Open(filename, 'UPDATE')
        d = f.GetDirectory(dir)
        if not d:
            d = make_root_dir(f, dir)
        d.cd()
        obj.Write()

    @staticmethod
    def parse_extension_from_file_name(name):
        ext = name.split('.')[-1]
        if ext is name:
            return None
        return ext

    def set_directory(self, directory):
        self.__check_and_create_directory(directory)
        self.dir = directory
