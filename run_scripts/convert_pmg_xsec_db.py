import argparse
import sys
from PyAnalysisTools.base.YAMLHandle import YAMLLoader, YAMLDumper
from PyAnalysisTools.AnalysisTools.XSHandle import Dataset


def main(argv):
    parser = argparse.ArgumentParser(description='Converter from PMG xsec DB to dataset format')
    parser.add_argument('input_file', help="PMG input file")
    parser.add_argument('--output_file', '-o', help='output file name')
    parser.add_argument('--dataset_decoration', '-ds', help="dataset decoration file")

    args = parser.parse_args()
    dataset_decoration = YAMLLoader.read_yaml(args.dataset_decoration)
    datasets = {}
    with open(args.input_file, 'r') as input_file:
        for line in input_file.readlines():
            try:
                ds_id, _, xsec, filter_eff, kfactor, _, _ = line.split()
            except ValueError:
                try:
                    ds_id, _, xsec, filter_eff, kfactor, _, _, _ = line.split()
                except ValueError:
                    continue
            if int(ds_id) not in dataset_decoration.keys():
                continue
            ds_id = int(ds_id)
            decoration = dataset_decoration[ds_id]
            if 'process_name' not in decoration:
                print ds_id
                continue
            dataset_info = {"is_mc": True,
                            "cross_section": float(xsec) / 1000.,
                            "dsid": ds_id,
                            "kfactor": float(kfactor),
                            "filtereff": float(filter_eff),
                            "latex_label": decoration['latex_label'] if 'latex_label' in decoration else None,
                            'process_name': decoration['process_name']}
            dataset = Dataset(**dataset_info)
            datasets[ds_id] = dataset

    YAMLDumper.dump_yaml(datasets, args.output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
