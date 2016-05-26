__author__ = 'marcusmorgenstern'
__mail__ = ''


from ROOT import TFile, TH1I


def prepare_cutflow_input():
    f = TFile.Open("CutflowTestInput.root", "RECREATE")
    cutflow_raw = TH1I("cutflow_raw", "cutflow_raw", 10, 0, 10)
    for i in range(10):
        cutflow_raw.SetBinContent(i, pow(10 - i, 2))
        cutflow_raw.GetXaxis().SetBinLabel(i + 1, "cut_%i" % i)
    f.cd()
    cutflow_raw.Write()
    f.Close()


def main():
    prepare_cutflow_input()


if __name__ == '__main__':
    main()
