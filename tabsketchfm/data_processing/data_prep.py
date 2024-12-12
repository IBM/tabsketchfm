import argparse
import bz2
import gzip
import json
import logging
import os
import random
import traceback
from enum import Enum

import chardet
import joblib
import numpy as np
import pandas as pd
import xxhash
from datasketch import MinHash
from pandas.io.parsers import TextFileReader
from dateutil.parser import parse
from sklearn.feature_extraction.text import HashingVectorizer


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# random.seed(0)
NROWS = 10000
DEBUG=False
vectorizer = HashingVectorizer(n_features=30000)

def _hash_func(d):
    return xxhash.xxh32(d).intdigest()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class DATA_TYPE(str, Enum):
    STRING = 'string'
    FLOAT = 'float'
    INTEGER = 'integer'
    DATE = 'date'

def read_table_from_original(file_name, metadata = None, dataset_type=None):
    try:
        if metadata is not None:
            with open(metadata) as f:
                md = json.load(f)
                md['file_name'] = file_name
        else:
            md = { "file_name": file_name }
        if file_name.endswith('bz2'):
            with bz2.open(file_name) as f:
                df = pd.read_csv(f, nrows=NROWS, on_bad_lines='skip')
        elif file_name.endswith('gz'):
            with gzip.open(file_name) as f:
                df = pd.read_csv(f, nrows=NROWS, on_bad_lines='skip')
        else:
            try:
                df = pd.read_csv(file_name, nrows = NROWS, on_bad_lines='skip', chunksize=NROWS)
                if isinstance(df, TextFileReader) and df is not None:
                    df = pd.concat(df, ignore_index=True)
            except:
                try:
                    df = pd.read_csv(file_name, nrows=NROWS, on_bad_lines='skip', engine='python')
                except:
                    with open(file_name, errors = 'backslashreplace') as f:    
                        try:
                            df = pd.read_csv(f, nrows=NROWS, encoding_errors='replace', sep=None, on_bad_lines='skip')
                        except:
                            df = pd.read_excel(file_name, nrows=NROWS)
        if df is not None:
            df.drop(['row_index'], axis=1, errors='ignore', inplace=True)
        return [(df, md)]

    except Exception as error:
        logger.exception(error)
        return None

def get_types(df):
    col_types = {}
    for col in df.columns:
        tp = pd.api.types.infer_dtype(df[col])
        if tp == 'string':
            try:
                dt = [parse(v) if v else None for v in df[col]]
                df[col + "_DATE"] = dt
                tp = DATA_TYPE.DATE
                col_types[col + "_DATE"] = tp
            except:
                pass
        if tp == 'integer':
            tp = DATA_TYPE.INTEGER
        elif tp == 'floating' or tp == 'decimal' or tp == 'mixed-integer-float':
            tp = DATA_TYPE.FLOAT
        time_types = ['datetime64', 'datetime', 'date', 'timedelta64', 'timedelta', 'time', 'period']
        other_types = ['bytes', 'mixed-integer', 'complex', 'categorical', 'boolean', 'mixed', 'unknown-array']
        if tp in time_types:
            tp = DATA_TYPE.DATE
        elif tp in other_types:
            tp = DATA_TYPE.STRING

        col_types[col] = tp
    #print(col_types)
    return df, col_types

def get_encoding(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']

def sanitize_column_names(col_names):
    new_names = []
    for name in col_names:
        name = name.strip('\'" ').replace('\n', '').replace(',', '').lower()
        new_names.append(name)

    return new_names

def check_consecutive(l):
    return sorted(l) == list(range(min(l), max(l) + 1))


def prep_data(filename, outpath, metadata=None, dataset_type=None, num_augs=3, obscure_columns=None, hashfunc=_hash_func, random_seed=0):
    random.seed(random_seed)
    try:
        df_list = read_table_from_original(filename, metadata, dataset_type)

        for df_pair in df_list:
            file_name = df_pair[1]['file_name']
            df = df_pair[0]
            if df is None or len(df) < 5:
                print('skipping because this is a very small file', filename)
                continue
            if DEBUG:
                print("starting " + file_name + "...")

            df, types = get_types(df)
            cols = preprocess_cols(df, types, obscure_columns, hashfunc=hashfunc, random_seed=random_seed)

            num_augs = int(num_augs)
            for i in range(num_augs):
                readable_hash, result = process_df(cols, df, df_pair[1], i, hashfunc=hashfunc, random_seed=random_seed)

                if result and len(result['columns']) > 0:
                    if os.path.exists(os.path.join(outpath, str(readable_hash) + '.json.bz2')):
                        print('duplicate', str(readable_hash) + '.json.bz2', filename)
                        csv = os.path.basename(filename).replace('.csv', '')
                        f = bz2.open(os.path.join(outpath, str(readable_hash) + csv + '.json.bz2'), 'w')
                    else:
                        print("writing :" + os.path.join(outpath, str(readable_hash) + '.json.bz2'))
                        f = bz2.open(os.path.join(outpath, str(readable_hash) + '.json.bz2'), 'w')
                    json_bytes = json.dumps(result, indent=4, cls=NpEncoder).encode('utf-8')
                    f.write(json_bytes)
                    f.flush()
                    f.close()
                    print("succeeded :" + os.path.join(outpath, str(readable_hash) + '.json.bz2'))
                    print("succeeded CSV:", filename, metadata)
                else:
                    print("!!!skipping columns")

    except:
        print(traceback.format_exc())
        print("error writing " + str(filename))


def transform_data():
    parser = argparse.ArgumentParser(add_help=False)
    # stored pipeline plan
    parser.add_argument("--input",  help='single input file / zip/ csv/ xls')
    parser.add_argument("--output", default='JSON file to write to')
    parser.add_argument("--dataset_type", help='ckan socrata or neither')
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--num_augs", default=3)
    parser.add_argument("--obscure_columns", default=None)

    args = parser.parse_args()
    prep_data(args.input, args.output, args.metadata, args.dataset_type, args.num_augs, args.obscure_columns)


def preprocess_cols(df, types, obscure_columns, hashfunc=_hash_func, random_seed=0):
    all_df = {}
    cols = {}

    for idx, col in enumerate(df.columns):
        try:
            c = {}
            cols[col] = c
            if obscure_columns:
                c['name'] = 'col' + str(idx)
                print('obscuring columns')
            else:
                c['name'] = col
            c['type'] = types[col]
            df[col] = df[col].replace('', np.nan)
            num_na = df[col].isna().sum()
            all_na = False
            if num_na == len(df[col]):
                all_na = True
            df[col] = df[col].dropna()
            c['num_nan'] = int(num_na)
            c['unique'] = len(df[col].unique())

            all_df[col] = [str(v).encode('utf-8') for v in df[col] if v]
            if types[col] == DATA_TYPE.STRING:
                all_df[col + '_words'] = [x.encode('utf-8') for v in df[col] if v for x in str(v).split()]
                c['cell_width_bytes'] = float(len(np.asarray(df[col]).tobytes()) / len(df))
                if DEBUG:
                    print('finished cell width computation')
            else:
                if types[col] == DATA_TYPE.INTEGER and c['unique'] > (.9 * len(df)):
                    try:
                        row_index = np.arange(len(df))
                        corr = np.corrcoef(df[col], row_index)
                        if corr[1][0] > .99:
                            print('high correlation with row index')
                            #continue
                        if DEBUG:
                            print('correlation')
                    except:
                        print(traceback.format_exc())
                        pass
                if types[col] == DATA_TYPE.INTEGER or types[col] == DATA_TYPE.FLOAT:
                    if all_na or len(df[col]) <= 1 or c['unique'] == 1:
                        continue
                    data = pd.DataFrame(data={'col': df[col]})
                    q = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
                    quantile = data['col'].quantile(q)
                    ql = quantile.tolist()
                    ql.append(np.mean(data['col']))
                    if c['unique'] > 2:
                        ql.append(np.std(data['col']))
                    else:
                        ql.append(0)
                    ql.append(np.nanmin(data['col']))
                    ql.append(np.nanmax(data['col']))
                    c['quantile'] = ql

        except:
            print('failed on column ' + col)
            print(traceback.format_exc())

    # print(f"random seed1  {random_seed}")
    if isinstance(hashfunc, HashingVectorizer):
        l = []
        for c in all_df:
            m = [x.decode() for x in all_df[c]]
            l.append(' '.join(m))

        csr = vectorizer.fit_transform(l)
        for i, col in enumerate(all_df):
            if not col.endswith('_words'):
                if len(l[i]) > 1:
                    cols[col]['hv'] = csr[i].todense()

    else:
        l = list(all_df.values())
        minhashes = MinHash.bulk(l, hashfunc=hashfunc, num_perm=100, seed=random_seed)
        if DEBUG:
            print('minhash computed')
        # order of values is supposed to match order of inserted keys as per 3.7 Python
        for i, col in enumerate(all_df):
            if col.endswith('_words'):
                col = col.replace('_words', '')
                cols[col]['min-hash-words'] = minhashes[i].digest().tolist()
            else:
                cols[col]['min-hash-exact'] = minhashes[i].digest().tolist()

        if DEBUG:
            print('minhash hashed')

    return cols

def process_df(orig_cols, df, table_metadata, i, hashfunc=_hash_func, random_seed=0):
    columns = list(df.columns)
   
    if DEBUG:
        print('old columns', df.columns)
        readable_hash_old = joblib.hash(df)

    columns = [c for c in columns if c in orig_cols]
    if i > 1:
        random.shuffle(columns)
        df = df[columns]

    if DEBUG:
        print('shuffled columns', columns)
    if DEBUG:
        print('new columns', df.columns)
    result = {}
    readable_hash = joblib.hash(df)
    # if DEBUG:
    #     assert readable_hash_old != readable_hash
    result['table_metadata'] = table_metadata
    table_metadata['rows'] = len(df)

    cols = {}
    for col in columns:
        cols[col] = orig_cols[col]

    result['columns'] = cols

    result['content_snapshot'] = create_content_snapshot(df, DEBUG, hashfunc=hashfunc, random_seed=random_seed)

    if DEBUG:
        print('content snapshot hashed')
    if len(cols) == 0:
        print('no columns found somehow to track')

    return readable_hash, result

def create_content_snapshot(df, DEBUG, hashfunc=_hash_func, random_seed=0):
    if DEBUG:
        print('computing rows')

    rows = [' '.join(val) for val in df.astype(str).values.tolist()]

    all_rows = [k.encode('utf-8') for k in rows]
    
    if DEBUG:
        print('rows computed')

    if isinstance(hashfunc, HashingVectorizer):
        content_snapshot = []
    else:
        # print(f"random seed {random_seed}")
        l = [all_rows]
        minhashes = MinHash.bulk(l, hashfunc=hashfunc, num_perm=100, seed=random_seed)
        content_snapshot = minhashes[0].digest().tolist()
    return content_snapshot


if __name__ == "__main__":
    transform_data()
