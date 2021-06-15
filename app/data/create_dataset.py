import errno
import os
import sys
import argparse
import re
import io

from pathlib import Path
import pandas as pd


def iterstream(iterable, buffer_size=io.DEFAULT_BUFFER_SIZE):
    class IterStream(io.RawIOBase):
        def __init__(self):
            self.leftover = None

        def readable(self):
            return True

        def readinto(self, b):
            try:
                length = len(b)
                chunk = self.leftover or next(iterable)
                output, self.leftover = chunk[:length], chunk[length:]
                b[:len(output)] = output
                return len(output)
            except StopIteration:
                return 0
    return io.BufferedReader(IterStream(), buffer_size=buffer_size)


class DataFile(object):
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename, 'r') as f:
            out_line = ""
            for line in f:
                line = line.rstrip()
                m = re.match(r"^[^\s+]", line)
                if m is not None:
                    yield f"{out_line}\n".encode('utf-8')
                    out_line = ""
                out_line += line
            if out_line:
                yield f"{out_line}\n".encode('utf-8')


def create_dataset(input_name, protdist):
    input_filename = f"/input/{input_name}"
    if Path(input_filename).is_file():
        filename = Path(input_name).name

        hdf_key = 'input_data'
        hdf_filename = f'/output/{filename}.h5'

        with pd.HDFStore(hdf_filename) as store:
            if protdist:
                converted_file = iterstream(DataFile(input_filename).read())
            for chunk in pd.read_csv(
                    input_filename if not protdist else converted_file,
                    sep="\s+",
                    index_col=0,
                    header=None,
                    skiprows=1,
                    chunksize=1000):
                store.put(hdf_key, chunk, format='t', index=False)

            store.create_table_index(hdf_key, optlevel=9, kind='full')

    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in', help='Input csv file name', required=True)
    ap.add_argument('-p', '--protdist', dest='protdist', help='Add this parameter if you are'
                                                              ' using protdist distance matrix',
                    action='store_true')
    ap.set_defaults(protdist=False)

    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(1)

    args = vars(ap.parse_args())

    input_name = args['in']
    protdist = args['protdist']

    create_dataset(input_name, protdist)
