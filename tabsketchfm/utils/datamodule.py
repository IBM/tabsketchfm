import bz2
import json
import random
from os import listdir
from os.path import isfile, join

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from tabsketchfm import TableSimilarityDataset, TabularDataset


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, dataset, pad_to_max_length,
                preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                train_batch_size, val_batch_size, dataloader_num_workers, run_on_sample, sample_size, concat=False, cols_equal=False, preprocessed_data=False, drop_last=True, shuffle=True, send_idx=False):
        super().__init__()
        self.data_dir = data_dir
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataset = dataset
        self.run_on_sample = run_on_sample
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.send_idx = send_idx

    def setup(self, stage):
        with bz2.open(self.dataset) as f:
            self.data_splits = json.load(f)
        if self.shuffle:
            random.shuffle(self.data_splits['train'])

    def pick_sample_files(self, split):
        # Use this just for local testing
        return self.data_splits[split][:self.sample_size]

    def train_dataloader(self):
        train_files = self.data_splits['train']
        if self.run_on_sample:
            train_files = self.pick_sample_files('train')
        print('Number of training samples: ', len(train_files))
        train_dataset = TabularDataset(data_dir=self.data_dir, files=train_files, transform= self.tokenizer.tokenize_function, shuffle = self.shuffle, send_idx = self.send_idx)
        return DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, drop_last=self.drop_last, num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        valid_files = self.data_splits['valid']
        if self.run_on_sample:
            valid_files = self.pick_sample_files('valid')
        print('Number of validation samples: ', len(valid_files))
        valid_dataset = TabularDataset(data_dir=self.data_dir, files=valid_files, transform=self.tokenizer.tokenize_function, shuffle = self.shuffle,  send_idx = self.send_idx)
        return DataLoader(dataset=valid_dataset, batch_size=self.val_batch_size, drop_last=self.drop_last, num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        test_files = self.data_splits['test']
        if self.run_on_sample:
            test_files = self.pick_sample_files('test')
        test_dataset = TabularDataset(data_dir=self.data_dir,files=test_files, transform=self.tokenizer.tokenize_function, shuffle = self.shuffle,  send_idx = self.send_idx)
        return DataLoader(dataset=test_dataset, batch_size=self.train_batch_size, drop_last=self.drop_last,
                        num_workers=self.dataloader_num_workers)



class FinetuneDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, dataset, pad_to_max_length,
                preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                train_batch_size, val_batch_size, dataloader_num_workers, run_on_sample, sample_size, concat=False, cols_equal=False,
                preprocessed_data = True, drop_last=True, shuffle=True, send_idx=False):
        super().__init__()
        self.data_dir = data_dir
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.run_on_sample = run_on_sample
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.concat = concat
        self.cols_equal = cols_equal
        self.preprocessed_data = preprocessed_data
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.send_idx = send_idx
        print(f'Setting preprocessed_data: {preprocessed_data}')

    def setup(self, stage):
        with open(self.dataset) as f:
            self.data_splits = json.load(f)
        print(self.dataset)
        print(len(self.data_splits))
        if self.shuffle:
            random.shuffle(self.data_splits['train'])
            random.shuffle(self.data_splits['valid'])
            random.shuffle(self.data_splits['test'])

    def train_dataloader(self):
        train_files = self.data_splits['train']
        if self.run_on_sample:
            train_files = self.pick_sample_files()
        print('Number of training samples: ', len(train_files))
        train_dataset = TableSimilarityDataset(data_dir=self.data_dir,
                                            table_similarity=train_files, transform=self.tokenizer.tokenize_function,
                                            concat=self.concat, cols_equal=self.cols_equal,
                                            preprocessed_data = self.preprocessed_data)
        return DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, drop_last=self.drop_last,
                        shuffle=self.shuffle, num_workers=self.dataloader_num_workers)

    def pick_sample_files(self):
        mypath = self.data_dir
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        random.shuffle(onlyfiles)
        # sample_files = onlyfiles[:self.sample_size]
        sample_files = onlyfiles
        to_return = []
        for table_entry in self.data_splits['train']:
            if table_entry['table1']['filename'] in sample_files and table_entry['table2']['filename'] in sample_files:
                to_return.append(table_entry)
        for table_entry in self.data_splits['valid']:
            if table_entry['table1']['filename'] in sample_files and table_entry['table2']['filename'] in sample_files:
                to_return.append(table_entry)

        return to_return[:self.sample_size]

    def val_dataloader(self):
        # num_train_samples = int(len(self.data_splits['train'])*0.9)
        valid_files = self.data_splits['valid']#[num_train_samples:]
        if self.run_on_sample:
            valid_files = self.pick_sample_files()
        print('Number of validation samples: ', len(valid_files))
        valid_dataset = TableSimilarityDataset(data_dir=self.data_dir,
                                            table_similarity=valid_files, transform=self.tokenizer.tokenize_function,
                                            cols_equal=self.cols_equal, concat=self.concat, preprocessed_data = self.preprocessed_data)
        return DataLoader(dataset=valid_dataset, shuffle = self.shuffle, batch_size=self.val_batch_size, drop_last=self.drop_last, num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        test_files = self.data_splits['test']
        if self.run_on_sample:
            test_files = self.pick_sample_files()
        test_dataset = TableSimilarityDataset(data_dir=self.data_dir,
                                            table_similarity=test_files, transform=self.tokenizer.tokenize_function,
                                            cols_equal=self.cols_equal, concat=self.concat, preprocessed_data = self.preprocessed_data)
        return DataLoader(dataset=test_dataset, shuffle = self.shuffle, batch_size=self.train_batch_size, drop_last=self.drop_last,
                        num_workers=self.dataloader_num_workers)
