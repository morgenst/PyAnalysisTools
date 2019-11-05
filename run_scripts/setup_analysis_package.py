from __future__ import print_function
from builtins import map
from builtins import object
import argparse
import os
from PyAnalysisTools.base.ShellUtils import make_dirs, copy


class ModuleCreator(object):
    """
    Class to setup a new analysis module
    """
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.path = kwargs['path']
        if kwargs['short_name'] is not None:
            self.short_name = kwargs['short_name']
        else:
            self.short_name = self.name
        self.modules = ['OutputModule', 'MainSelectionModule']
        self.module_path = os.path.abspath(os.path.join(self.path, self.name))

    def setup_directory(self):
        """
        Create package directory if not existing
        :return: nothing
        :rtype: None
        """
        make_dirs(self.module_path)
        folders = ['macros', 'Root', 'data', 'run', 'run/configs', self.name]
        list(map(lambda f: make_dirs(os.path.join(self.module_path, f)), folders))

    def add_output_module(self):
        with open(os.path.join(self.module_path, 'Root', 'OutputModule.cxx'), 'w') as f:
            print('#include <{:s}/OutputModule.h>\n\n'.format(self.name), file=f)
            print('void OutputModule::initialize (){\n', file=f)
            print('\tstd::function<float()> fYourFctPtr = [=](){ return 1111.; /*enter calculation here*/}; \n', file=f)
            print('\tcore()->AddTreeBranch("your_branch_name", fYourFctPtr, "your_tree_name");', file=f)
            print('}', file=f)
        with open(os.path.join(self.module_path, self.name, 'OutputModule.h'), 'w') as f:
            print('#pragma once\n', file=f)
            print('#include <ELCore/Module.h>\n', file=f)
            print('class OutputModule : public Module{', file=f)
            print('\tpublic:', file=f)
            print('\tOutputModule(const std::string& moduleName) : Module(moduleName) {}', file=f)
            print('\tvirtual ~OutputModule(){}\n', file=f)
            print('\tvirtual void initialize();\n', file=f)
            print('\t//configurables', file=f)
            print('\tstd::string exampleKey = "kYourCollection";\n', file=f)
            print('\tprivate: \n', file=f)
            print('\tClassDef(OutputModule, 1);', file=f)
            print('};', file=f)

    def add_mainselection_module(self):
        with open(os.path.join(self.module_path, 'Root', 'MainSelectionModule.cxx'), 'w') as f:
            print('#include <{:s}/MainSelectionModule.h>\n\n'.format(self.name), file=f)
            print('void MainSelectionModule::initialize (){\n', file=f)
            print('\tcore()->addTemplate("SOME_CUR", &MainSelectionModule::someCut, this); \n', file=f)
            print('}\n', file=f)
            print('void MainSelectionModule::someCut() {', file=f)
            print('\t//Do stuff here, e.g. print configurable', file=f)
            print('\tlog->info("Example configurable {}", someConfigurable);', file=f)
            print('}\n', file=f)
        with open(os.path.join(self.module_path, self.name, 'MainSelectionModule.h'), 'w') as f:
            print('#pragma once\n', file=f)
            print('#include <ELCore/Module.h>', file=f)
            print('#include <ELCore/Exceptions.h>', file=f)
            print('#include <ELCore/Container.h>\n', file=f)
            print('\nnamespace CP {', file=f)
            print('\tclass someCPTool;', file=f)
            print('}\n', file=f)
            print('class MainSelectionModule : public Module {', file=f)
            print('\tpublic:', file=f)
            print('\tMainSelectionModule() : Module("MainSelectionModule") {}', file=f)
            print('\tvirtual ~MainSelectionModule(){}\n', file=f)
            print('\tvirtual void initialize();\n', file=f)
            print('\t//configurables', file=f)
            print('\tstd::string someConfigurable = "I am a configurable variable to be set in module_config.yml";\n',
                  file=f)
            print('\tprivate:', file=f)
            print('// DO NOT FORGET TO DISABLE THE ROOT STREAMERS VIA //!', file=f)
            print('\tstd::shared_ptr<CP::someCPTool> m_myCPTool; //!', file=f)
            print('\tvoid someCut();\n', file=f)
            print('\tClassDef(MainSelectionModule, 1);', file=f)
            print('};', file=f)

    def add_readme(self):
        with open(os.path.join(self.module_path, 'README.md'), 'w') as f:
            print('===========================', file=f)
            print('{:s}'.format(self.name), file=f)
            print('===========================\n\n', file=f)
            print('Dependencies', file=f)
            print('------------', file=f)
            print('* ELCore', file=f)
            print('* ELExtrapolator', file=f)
            print('* LumiCalculator', file=f)
            print('* PyAnalysisTools\n\n', file=f)
            print('Setup', file=f)
            print('-----', file=f)
            print('For details on the ELBrain framework visit: https://elbraindocs.web.cern.ch \n\n', file=f)
            print('XXX Analysis', file=f)
            print('------------', file=f)
            print('Specify here details on this package', file=f)

    def add_run_script(self):
        with open(os.path.join(self.module_path, 'run', 'run.py'), 'w') as f:
            print('from ELCore import RunManager \n', file=f)
            print('if __name__ == "__main__":', file=f)
            print('\tparser = RunManager.get_parser(\'{:s}\')'.format(self.name), file=f)
            print('\targs = parser.parse_args()', file=f)
            print('\tmanager = RunManager.RunManager("{:s}", abbrev="{:s}_", **vars(args))'.format(self.name,
                                                                                                   self.short_name),
                  file=f)
            print('\tmanager.run()', file=f)

    def add_link_def(self):
        with open(os.path.join(self.module_path, 'Root', 'LinkDef.h'), 'w') as f:
            for mod in self.modules:
                print('#include <{:s}/{:s}.h>'.format(self.name, mod), file=f)
            print('#ifdef __CINT__', file=f)
            print('#pragma link off all globals;', file=f)
            print('#pragma link off all classes;', file=f)
            print('#pragma link off all functions;', file=f)
            print('#pragma link C++ nestedclass;\n', file=f)
            print('#endif\n', file=f)
            print('#ifdef __CINT__', file=f)
            for mod in self.modules:
                print('#pragma link C++ class {:s}+;'.format(mod), file=f)
            print('#endif', file=f)

    def add_cmake_file(self):
        with open(os.path.join(self.module_path, 'CMakeLists.txt'), 'w') as f:
            print('################################################################################', file=f)
            print('# Package: {:s}'.format(self.name), file=f)
            print('################################################################################\n', file=f)
            print('atlas_subdir({:s}) \n'.format(self.name), file=f)

            print('set(CMAKE_CXX_FLAGS "-std=c++14") \n', file=f)
            print('# Declare the package\'s dependencies:', file=f)
            print('atlas_depends_on_subdirs(\n \t\tPUBLIC', file=f)
            print('\t\t\tELCore', file=f)
            print('\t\t\tELCommon', file=f)
            print('\t\t\tLumiCalculator', file=f)
            print('\t\t\t${extra_deps} )\n', file=f)
            print('# External dependencies:', file=f)
            print('find_package(Boost)', file=f)
            print('find_package(ROOT', file=f)
            print('\t\tCOMPONENTS;', file=f)
            print('\t\tCore', file=f)
            print('\t\tTree', file=f)
            print('\t\tMathCore', file=f)
            print('\t\tHist', file=f)
            print('\t\tRIO )\n', file=f)
            print('# Libraries in the package:', file=f)
            print('atlas_add_root_dictionary({:s}Lib'.format(self.name), file=f)
            print('\t\t{:s}LibCintDict'.format(self.name), file=f)
            print('\t\tROOT_HEADERS', file=f)
            print('\t\t{:s}/*.h'.format(self.name), file=f)
            print('\t\tRoot/LinkDef.h', file=f)
            print('\t\tEXTERNAL_PACKAGES ROOT )\n', file=f)
            print('atlas_add_library({:s}Lib'.format(self.name), file=f)
            print('\t{:s}/*.h Root/*.cxx ${{{:s}LibCintDict}}'.format(self.name, self.name), file=f)
            print('\tPUBLIC_HEADERS {:s}'.format(self.name), file=f)
            print('\tINCLUDE_DIRS ${ROOT_INCLUDE_DIRS} ${BOOST_INCLUDE_DIRS}', file=f)
            print('\tLINK_LIBRARIES ELCoreLib ELCommonLib LumiCalculatorLib ${ROOT_LIBRARIES} ${BOOST_LIBRARIES}'
                  '${extra_libs} )\n', file=f)
            print('#Install files from the package:', file=f)
            print('atlas_install_data(data/*)', file=f)

    def copy_and_update_package_setup(self):
        with open(os.path.join(self.path, '.setup_package.py'), 'w') as f_out:
            with open(os.path.join(os.path.dirname(__file__), '.common_setup_package.py'), 'r') as f_in:
                for line in f_in.readline():
                    line = line.replace('PACKAGE_NAME_LOWER', self.name.lower())
                    line = line.replace('PACKAGE_NAME_ABBR', self.short_name)
                    line = line.replace('PACKAGE_NAME', self.name)
                    print(line, file=f_out)

    def copy_common_files(self):
        print('COPY', os.path.join('.analysis_package_generic', 'common_setup.sh'), os.path.join(self.path, 'setup.sh'))
        copy(os.path.join(os.path.dirname(__file__), '.analysis_package_generic', 'common_setup.sh'),
             os.path.join(self.path, self.module_path, 'setup.sh'))

    def create(self):
        self.setup_directory()
        self.add_link_def()
        self.add_cmake_file()
        self.add_readme()
        self.add_run_script()
        self.add_output_module()
        self.add_mainselection_module()
        self.copy_and_update_package_setup()
        self.copy_common_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup new analysis package")
    parser.add_argument('name', help="Name of new package")
    parser.add_argument('--short_name', '-a', default=None, help="Abbreviation of package name")
    parser.add_argument('--path', '-p', default='../../',
                        help="path where the package should be set up (default ELCore/../)")

    args = parser.parse_args()
    creator = ModuleCreator(**vars(args))
    creator.create()
