import argparse
import os
import sys

from PyAnalysisTools.base.YAMLHandle import YAMLLoader


def main(argv):
    parser = argparse.ArgumentParser(description="Synchronisation script")
    parser.add_argument("path", type=str, help="input path")
    parser.add_argument("--config", "-c", type=str, required=True, help="configuration")

    args = parser.parse_args()

    config = YAMLLoader.read_yaml(args.config)
    for data in config["content"]:
        cmd = ["rsync", "-ravu", os.path.join(args.path, data, "."), " %s@%s:%s" % (config["user"],
                                                                                    config["server"],
                                                                                    os.path.join(config["destination"],
                                                                                                 data))]
        os.system(" ".join(cmd))


if __name__ == '__main__':
    main(sys.argv)
