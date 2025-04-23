import bz2
import json
import os
import random
from os.path import isfile, join

import torch
from torch.utils.data import Dataset

from .data_prep import (get_types, preprocess_cols, process_df,
                        read_table_from_original)


class TabularDataset(Dataset):
    def __init__(self, data_dir, files, transform, shuffle = True, send_idx = False):
        self.data_dir = data_dir
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(files)
        self.files = files
        self.transform = transform
        self.send_idx = send_idx
        print('Number of files:' + str(len(self.files)))
        
    def __len__(self):
        return len(self.files)

    def find_existing_file(self):
        found = False
        f = None
        while not found:
            f = random.sample(self.files, 1)[0]
            if os.path.exists(os.path.join(self.data_dir, f)):
                found = True
        return f

    def __getitem__(self, idx):
        d = self.files[idx]
        
        table = d['json']
        col_index = d['column']
        json_path = os.path.join(self.data_dir, table)

        # the latest runs seem to produce different output for data prep which needs to be
        # looked into.
        if not os.path.exists(json_path):
            print('warning: cannot find file:', json_path)
            json_path = os.path.join(self.data_dir, self.find_existing_file())
        
        with bz2.open(json_path) as f:
            data = json.load(f)
        if self.send_idx:
            return idx, self.transform(data, col_index)
        else:
            return self.transform(data, col_index)




class TableSimilarityDataset(Dataset):
    def __init__(self, data_dir, table_similarity, transform, concat=True, cols_equal=False, max_pos_embeddings=512, preprocessed_data=True, send_idx=False, extract_embedding=False):
        self.data_dir = data_dir
        self.table_similarity = table_similarity
        self.transform = transform
        self.concat = concat
        self.send_idx = send_idx
        self.cols_equal = cols_equal
        self.max_pos_embeddings = int(max_pos_embeddings / 2)
        self.extract_embedding = extract_embedding
        self.preprocessed_data = preprocessed_data
        print('Number of files:' + str(len(self.table_similarity)))
        self.csv_fname_to_df_hash_name = {}
        if self.preprocessed_data:
            cache_fname = self.data_dir + '/csv_fname_to_df_hash_name.json'
            if isfile(cache_fname):
                self.csv_fname_to_df_hash_name = json.load(open(cache_fname))
                print(f'Loaded {len(self.csv_fname_to_df_hash_name)} files from {cache_fname}')
            else:
                onlyfiles = [join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))]
                for fname_tab in onlyfiles: #TODO: should probably save that list in the same directory!!
                    # print('Reading: ', fname_tab)
                    if not fname_tab.endswith('json.bz2'):
                        continue
                    with bz2.open(fname_tab) as f:
                        table = json.load(f)
                        self.csv_fname_to_df_hash_name[table["table_metadata"]["file_name"].split('/')[-1]] = fname_tab
                # print('Built csv_fname_to_df_hash_name: ', len(self.csv_fname_to_df_hash_name))
                # with open(cache_fname, 'w') as cache_file:
                #     json.dump(self.csv_fname_to_df_hash_name, cache_file, indent=4)

            self.table_similarity = []
            num_skipped_examples = 0
            uniq_files = set([])
            for idx in range(len(table_similarity)):
                d = table_similarity[idx]
                if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name and d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                    self.table_similarity.append(d)
                else:
                    num_skipped_examples += 1
                if d['table1']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    print('Could not find ', d['table1']['filename'], ' -- base name: ', d['table1']['filename'].split('/')[-1])
                    uniq_files.add(d['table1']['filename'])
                if d['table2']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    uniq_files.add(d['table2']['filename'])

            print('num_skipped_examples: ', num_skipped_examples)
            print('uniq_files not found: ', len(uniq_files))
            for f in uniq_files:
                print(f)
            print('*'*20)
            print('Number of files (filtered):' + str(len(self.table_similarity)))

    def __len__(self):
        return len(self.table_similarity)

    def process_table(self, filename):
        df_list = read_table_from_original(filename)
        df_pair = df_list[0]
        file_name = df_pair[1]['file_name']
        df = df_pair[0]
        df, types = get_types(df)
        cols = preprocess_cols(df, types, True)

        readable_hash, result = process_df(cols, df, table_metadata={})
        return result
    
    def __getitem__(self, idx):
        d = self.table_similarity[idx]
        if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
            fname_tab1 = self.csv_fname_to_df_hash_name[d['table1']['filename'].split('/')[-1]]
        else:
            fname_tab1 = None
        if self.preprocessed_data and fname_tab1 and os.path.isfile(fname_tab1):
            # print('Found preprocessed name: ', fname_tab1)
            with bz2.open(fname_tab1) as f:
                table_1 = json.load(f)
        else:
            # print('process_table: ', fname_tab1)
            table_1 = self.process_table(os.path.join(self.data_dir, d['table1']['filename']))
        if d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
            fname_tab2 = self.csv_fname_to_df_hash_name[d['table2']['filename'].split('/')[-1]]
        else:
            fname_tab2 = None
        if self.preprocessed_data and fname_tab2 and os.path.isfile(fname_tab2):
            # print('Found preprocessed name: ', fname_tab2)
            with bz2.open(fname_tab2) as f:
                table_2 = json.load(f)
        else:
            # print('process_table: ', fname_tab1)
            table_2 = self.process_table(os.path.join(self.data_dir, d['table2']['filename']))

        label = d['label']
        if isinstance(label, list):
            target = torch.zeros(d['num_classes'])
            target[label] = 1.
            label = target
        data1 = self.transform(table_1)
        data2 = self.transform(table_2)
#        print(f"data shape {data1.shape} {data2.shape}")
        col_info = sorted(table_1['columns'].values(), key=lambda x: x['name']) #TODO: do we need sorting here
        t1cols = [c['name'] for c in col_info]
        col_info = sorted(table_2['columns'].values(), key=lambda x: x['name'])
        t2cols = [c['name'] for c in col_info]
        if self.cols_equal:
            assert t1cols == t2cols
            assert torch.equal(data1['input_ids'], data2['input_ids']), str([c['name'] for c in table_1['columns']]) + str([c['name'] for c in table_2['columns']])

        data = {}
        for key in data1:
            if self.concat:
                # at a high level, what is happening here is table1 content snapshot SEP table2 content snapshot SEP
                # all cols from table 1 SEP all cols from table 2
                # key 2 is values associated with a separator, table name in fine tuning is always omitted
                # values differ in terms of they are just integers (col type or col pos) or they are vectors (quantiles, minhashes)
                try_complex_concat = True
                if try_complex_concat:
                    padding = data1[key][2]
                    if len(padding.shape) == 0:
                        padding = padding.reshape(1)
                    else:
                        padding = torch.unsqueeze(padding, dim=0)
                    # stack CLS (key 0), table 1 name (key 1), SEP (key2), table 2 name (key 1), SEP (key 2) 
                    z = torch.stack((data1[key][0], data1[key][1], data1[key][2], data2[key][1], data2[key][2]))
                    table1_start = z.size()[0]
                    table1_end = torch.concat((z, data1[key][3:self.max_pos_embeddings])).size()[0]
                    table2_start = torch.concat((z, data1[key][3:self.max_pos_embeddings], padding)).size()[0]
                    data[key] = torch.concat((z, data1[key][3:self.max_pos_embeddings], padding, data2[key][3:self.max_pos_embeddings]))
                else:
                    # old style call: table1, all columns table2, all columns """
                    data[key] = torch.cat((data1[key][:self.max_pos_embeddings], data2[key][:self.max_pos_embeddings]), 0)
            else:
                data[key] = (data1[key], data2[key])

        if self.extract_embedding:
            # print("Extracting embedding ==== ", type(label), type(idx), type(len(t1cols)), type(len(t2cols)), type(table1_start), type(table1_end), type(table2_start), "====")
            return data, label, idx, len(t1cols), len(t2cols), table1_start, table1_end, table2_start
        elif self.send_idx:
            return data, label, idx 
        else:
            return data, label 
        

class TableColumnSearchDataset(TableSimilarityDataset):
    def __init__(self, data_dir, search_dataset, query_tables, transform, concat=True, cols_equal=False, max_pos_embeddings=512,
                 preprocessed_data=True, topK=100):
        self.data_dir = data_dir
        self.search_dataset = search_dataset
        self.transform = transform
        self.concat = concat
        self.cols_equal = cols_equal
        self.max_pos_embeddings = int(max_pos_embeddings / 2)
        self.preprocessed_data = preprocessed_data
        print('Number of queries:' + str(len(self.search_dataset)))
        self.csv_fname_to_df_hash_name = {}
        self.search_query_data = []
        self.topK = topK
        num_skipped_examples = 0
        uniq_files = set([])
        cache_fname = self.data_dir+'/csv_fname_to_df_hash_name.json'

        if isfile(cache_fname):
            self.csv_fname_to_df_hash_name = json.load(open(cache_fname))
            print(f'Loaded {len(self.csv_fname_to_df_hash_name)} files from {cache_fname}')
        else:
            onlyfiles = [join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))]
            for fname_tab in onlyfiles: #TODO: should probably save that list in the same directory!!
                # print('Reading: ', fname_tab)
                if not fname_tab.endswith('json.bz2'):
                    continue
                with bz2.open(fname_tab) as f:
                    table = json.load(f)
                    self.csv_fname_to_df_hash_name[table["table_metadata"]["file_name"].split('/')[-1]] = fname_tab
            print('Built csv_fname_to_df_hash_name: ', len(self.csv_fname_to_df_hash_name))
            # with open(cache_fname, 'w') as cache_file:
            #     json.dump(self.csv_fname_to_df_hash_name, cache_file, indent=4)

        
        for query_column in query_tables:
            query = query_column.split(":")[0]+".csv"
            if query_column not in search_dataset.keys():
                print(f"Skipping query {query}")
                continue
            results = search_dataset[query_column]
            query_data = []
            for r, response_column in enumerate(results):
                response = response_column.split(":")[0]+".csv"
                if r > self.topK:
                    break
                d = {'table1': {'filename': query},
                     'table2': {'filename': response}}
                if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name and d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                    query_data.append(d)
                else:
                    if d['table2']['filename'].split('/')[-1].split(".")[0] +".csv" in self.csv_fname_to_df_hash_name:
                        prefix = "/".join(d['table2']['filename'].split('/')[:-1])
                        filename  = d['table2']['filename'].split('/')[-1].split(".")[0] + ".csv"
                        d['table2']['filename'] = prefix+"/"+filename
                        query_data.append(d)
                    else:
                        num_skipped_examples += 1
                if d['table1']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    print('Could not find query ', d['table1']['filename'], ' -- base name: ', d['table1']['filename'].split('/')[-1])
                    uniq_files.add(d['table1']['filename'])
                if d['table2']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    print('Could not find response file', d['table2']['filename'], ' -- query: ', d['table1']['filename'])
                    uniq_files.add(d['table2']['filename'])
            self.search_query_data.append(query_data) 

        print('num_skipped_examples: ', num_skipped_examples)
        print('uniq_files not found: ', len(uniq_files))
        for f in uniq_files:
            print(f)
        print('*'*20)
        print('Number of files (filtered):' + str(len(self.search_query_data)))

    def __len__(self):
        return len(self.search_query_data)

    
    def __getitem__(self, idx):
        batch_data = []
        for d in self.search_query_data[idx]:
            if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                fname_tab1 = self.csv_fname_to_df_hash_name[d['table1']['filename'].split('/')[-1]]
            else:
                fname_tab1 = None
            if self.preprocessed_data and fname_tab1 and os.path.isfile(fname_tab1):
                # print('Found preprocessed name: ', fname_tab1)
                with bz2.open(fname_tab1) as f:
                    table_1 = json.load(f)
            else:
                # print('process_table: ', fname_tab1)
                table_1 = self.process_table(os.path.join(self.data_dir, d['table1']['filename']))
            if d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                fname_tab2 = self.csv_fname_to_df_hash_name[d['table2']['filename'].split('/')[-1]]
            else:
                fname_tab2 = None
            if self.preprocessed_data and fname_tab2 and os.path.isfile(fname_tab2):
                # print('Found preprocessed name: ', fname_tab2)
                with bz2.open(fname_tab2) as f:
                    table_2 = json.load(f)
            else:
                # print('process_table: ', fname_tab1)
                table_2 = self.process_table(os.path.join(self.data_dir, d['table2']['filename']))

            
            data1 = self.transform(table_1)
            data2 = self.transform(table_2)
    #        print(f"data shape {data1.shape} {data2.shape}")
            col_info = sorted(table_1['columns'].values(), key=lambda x: x['name']) #TODO: do we need sorting here
            t1cols = [c['name'] for c in col_info]
            col_info = sorted(table_2['columns'].values(), key=lambda x: x['name'])
            t2cols = [c['name'] for c in col_info]
            if self.cols_equal:
                assert t1cols == t2cols
                assert torch.equal(data1['input_ids'], data2['input_ids']), str([c['name'] for c in table_1['columns']]) + str([c['name'] for c in table_2['columns']])

            data = {}
            for key in data1:
                padding = data1[key][2]
                if len(padding.shape) == 0:
                    padding = padding.reshape(1)
                else:
                    padding = torch.unsqueeze(padding, dim=0)
                # stack CLS (key 0), table 1 name (key 1), SEP (key2), table 2 name (key 1), SEP (key 2) 
                z = torch.stack((data1[key][0], data1[key][1], data1[key][2], data2[key][1], data2[key][2]))
                data[key] = torch.concat((z, data1[key][3:self.max_pos_embeddings], padding, data2[key][3:self.max_pos_embeddings]))
            batch_data.append(data)
        return torch.utils.data.default_collate(batch_data)

class TableSearchDataset(TableSimilarityDataset):
    def __init__(self, data_dir, search_dataset, query_tables, transform, concat=True, cols_equal=False, max_pos_embeddings=512,
                 preprocessed_data=True, topK=100):
        self.data_dir = data_dir
        self.search_dataset = search_dataset
        self.transform = transform
        self.concat = concat
        self.cols_equal = cols_equal
        self.max_pos_embeddings = int(max_pos_embeddings / 2)
        self.preprocessed_data = preprocessed_data
        print('Number of queries:' + str(len(self.search_dataset)))
        self.csv_fname_to_df_hash_name = {}
        self.search_query_data = []
        self.topK = topK
        num_skipped_examples = 0
        uniq_files = set([])
        cache_fname = self.data_dir+'/csv_fname_to_df_hash_name.json'

        if isfile(cache_fname):
            self.csv_fname_to_df_hash_name = json.load(open(cache_fname))
            print(f'Loaded {len(self.csv_fname_to_df_hash_name)} files from {cache_fname}')
        else:
            onlyfiles = [join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))]
            for fname_tab in onlyfiles: #TODO: should probably save that list in the same directory!!
                # print('Reading: ', fname_tab)
                if not fname_tab.endswith('json.bz2'):
                    continue
                with bz2.open(fname_tab) as f:
                    table = json.load(f)
                    self.csv_fname_to_df_hash_name[table["table_metadata"]["file_name"].split('/')[-1]] = fname_tab
            print('Built csv_fname_to_df_hash_name: ', len(self.csv_fname_to_df_hash_name))
            # with open(cache_fname, 'w') as cache_file:
            #     json.dump(self.csv_fname_to_df_hash_name, cache_file, indent=4)

        for query in query_tables:
            if query not in search_dataset.keys():
                print(f"Skipping query {query}")
                continue
            results = search_dataset[query]
            query_data = []
            for r, response in enumerate(results):
                if r > self.topK:
                    break
                d = {'table1': {'filename': query},
                     'table2': {'filename': response}}
                if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name and d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                    query_data.append(d)
                else:
                    if d['table2']['filename'].split('/')[-1].split(".")[0] +".csv" in self.csv_fname_to_df_hash_name:
                        prefix = "/".join(d['table2']['filename'].split('/')[:-1])
                        filename  = d['table2']['filename'].split('/')[-1].split(".")[0] + ".csv"
                        d['table2']['filename'] = prefix+"/"+filename
                        query_data.append(d)
                    else:
                        num_skipped_examples += 1
                if d['table1']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    print('Could not find query ', d['table1']['filename'], ' -- base name: ', d['table1']['filename'].split('/')[-1])
                    uniq_files.add(d['table1']['filename'])
                if d['table2']['filename'].split('/')[-1] not in self.csv_fname_to_df_hash_name:
                    print('Could not find response file', d['table2']['filename'], ' -- query: ', d['table1']['filename'])
                    uniq_files.add(d['table2']['filename'])
            self.search_query_data.append(query_data) 

        print('num_skipped_examples: ', num_skipped_examples)
        print('uniq_files not found: ', len(uniq_files))
        for f in uniq_files:
            print(f)
        print('*'*20)
        print('Number of files (filtered):' + str(len(self.search_query_data)))

    def __len__(self):
        return len(self.search_query_data)

    
    def __getitem__(self, idx):
        batch_data = []
        for d in self.search_query_data[idx]:
            if d['table1']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                fname_tab1 = self.csv_fname_to_df_hash_name[d['table1']['filename'].split('/')[-1]]
            else:
                fname_tab1 = None
            if self.preprocessed_data and fname_tab1 and os.path.isfile(fname_tab1):
                # print('Found preprocessed name: ', fname_tab1)
                with bz2.open(fname_tab1) as f:
                    table_1 = json.load(f)
            else:
                # print('process_table: ', fname_tab1)
                table_1 = self.process_table(os.path.join(self.data_dir, d['table1']['filename']))
            if d['table2']['filename'].split('/')[-1] in self.csv_fname_to_df_hash_name:
                fname_tab2 = self.csv_fname_to_df_hash_name[d['table2']['filename'].split('/')[-1]]
            else:
                fname_tab2 = None
            if self.preprocessed_data and fname_tab2 and os.path.isfile(fname_tab2):
                # print('Found preprocessed name: ', fname_tab2)
                with bz2.open(fname_tab2) as f:
                    table_2 = json.load(f)
            else:
                # print('process_table: ', fname_tab1)
                table_2 = self.process_table(os.path.join(self.data_dir, d['table2']['filename']))

            
            data1 = self.transform(table_1)
            data2 = self.transform(table_2)
    #        print(f"data shape {data1.shape} {data2.shape}")
            col_info = sorted(table_1['columns'].values(), key=lambda x: x['name']) #TODO: do we need sorting here
            t1cols = [c['name'] for c in col_info]
            col_info = sorted(table_2['columns'].values(), key=lambda x: x['name'])
            t2cols = [c['name'] for c in col_info]
            if self.cols_equal:
                assert t1cols == t2cols
                assert torch.equal(data1['input_ids'], data2['input_ids']), str([c['name'] for c in table_1['columns']]) + str([c['name'] for c in table_2['columns']])

            data = {}
            for key in data1:
                padding = data1[key][2]
                if len(padding.shape) == 0:
                    padding = padding.reshape(1)
                else:
                    padding = torch.unsqueeze(padding, dim=0)
                # stack CLS (key 0), table 1 name (key 1), SEP (key2), table 2 name (key 1), SEP (key 2) 
                z = torch.stack((data1[key][0], data1[key][1], data1[key][2], data2[key][1], data2[key][2]))
                data[key] = torch.concat((z, data1[key][3:self.max_pos_embeddings], padding, data2[key][3:self.max_pos_embeddings]))
            batch_data.append(data)
        return torch.utils.data.default_collate(batch_data)

