import datetime
import glob
import logging
import shutil

logging.disable(logging.CRITICAL)


def tearDownModule():
    folder_pattern = "output_{:s}*".format(datetime.date.today().strftime("%Y%m%d"))
    folders = glob.glob(folder_pattern)
    for fn in folders:
        try:
            shutil.rmtree(fn)
        except OSError:
            pass
    folders = glob.glob('*foo*')
    for fn in folders:
        try:
            shutil.rmtree(fn)
        except OSError:
            pass
