import argparse

from PyAnalysisTools.AnalysisTools import CutflowAnalyser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run script for cutflow analyser")
    parser.add_argument("filelist", "f", nargs="+", help="filelist")

    args = parser.parse_args()
    analyser = CutflowAnalyser(args.filelist)
    analyser.execute()


