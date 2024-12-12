import math

import numpy
import torch
from scipy import stats


def fake_tablename_metadata(example, tokenizer):
    desc = []
    m = example['table_metadata']
    if 'table_name' in m and m['table_name'] is not None and len(m['table_name'].strip()) > 0 :
        desc.append(m['table_name'])
    else:
        desc.append('table name')
    if 'table_description' in m and m['table_description'] is not None:
        desc.append(m['table_description'])
    if 'dataset_description' in m and m['dataset_description'] is not None:
        desc.append(m['dataset_description'])

    cols = [str(example['columns'][c]['name']) for c in example['columns']]
    str_cols = ' '.join(desc) + tokenizer.SEPARATOR + tokenizer.SEPARATOR.join(cols)
    return cols, str_cols

def get_table_metadata_open_data(example, tokenizer):
    desc = []
    m = example['table_metadata']
    if 'table_name' in m and m['table_name'] is not None:
        desc.append(m['table_name'])
    if 'table_description' in m and m['table_description'] is not None:
        desc.append(m['table_description'])
    if 'dataset_description' in m and m['dataset_description'] is not None:
        desc.append(m['dataset_description'])

    cols = [str(example['columns'][c]['name']) for c in example['columns']]
    str_cols = ' '.join(desc) + tokenizer.SEPARATOR + tokenizer.SEPARATOR.join(cols)
    return cols, str_cols

class Tokenizer(object):

    def __init__(self, tokenizer, config, table_metadata_func=get_table_metadata_open_data, embedding_extraction=False):
        self.tokenizer = tokenizer
        self.config = config
        self.SEPARATOR = self.tokenizer.sep_token
        self.SEPARATOR_ID = tokenizer.sep_token_id
        self.CLS_ID = tokenizer.cls_token_id
        self.MASK_ID = tokenizer.mask_token_id
        self.NOT_ID = len(tokenizer) - 1
        self.type_dict = {'string':1, 'integer':2, 'float':3, 'date': 4  }
        self.MINHASH_LENGTH = 100
        self.QUANTILE_LENGTH = 13
        self.CELL_WIDTH_BYTES = 1
        self.COUNT_NAN = 1
        self.COUNT_UNIQUE = 1
        self.MINHASH_SIZE = self.MINHASH_LENGTH * 2
        self.TOTAL = self.QUANTILE_LENGTH + self.CELL_WIDTH_BYTES + self.COUNT_NAN + self.COUNT_UNIQUE
        self.MLM_PROBABILITY=0.15  # masked language modeling probability for predicting the descriptions of tables
        self.TOTAL_MAX_COLS = 10   # total number of columns that will be masked, one at a time.  A random sample of max columns is masked for large tables.
        self.HIDDEN_SIZE = config.hidden_size
        self.PAD_VALUE = self.HIDDEN_SIZE - self.TOTAL
        self.MINHASH_PAD_VALUE = self.HIDDEN_SIZE - (2 * self.MINHASH_LENGTH)
        self.MAX_COL_TOKENS = 5
        self.COL_TOKEN_PROBABILITY = .6
        self.table_metadata_func = table_metadata_func
        self.embedding_extraction = embedding_extraction

    """
    Build the tokenized dataset as per requirement of OCP with minimal dependence on transformers
    The types of encoding added per column are:
    1.  Type of column (string/numeric/date)
    2.  Column position (useful sometimes because some columns reflect a single attribute split apart - e.g. first name, last name)
    3.  Token position within a column, so word piece position.  Note for both column position and token position, we encode -1 for special tokens
    4.  An encoding of a sample of values in the column:
            a.  For string columns, this is a minhash of exact values, plus minhash of their 'word' tokens - consider tokenizing using BERT - later?
            b.  For numeric columns, this is a minhash of exact values for joinability, plus quantile and mean/std.
            c.  Whether each value in the column is unique
    """
    def create_col_encodings(self, t):
        y = numpy.logical_or(numpy.logical_or(t == self.SEPARATOR_ID, t == self.CLS_ID), t == 0)
        x = numpy.where(y, 0, numpy.cumsum(t == self.SEPARATOR_ID) + 1)
        return x

    def mask_col(self, r, c, inputs, labels):
        # create a mask for all the tokens of a column name
        col_index = torch.from_numpy(numpy.where(c == r, 1, 0)).bool()
        real_tokens = torch.from_numpy(numpy.where(inputs != self.tokenizer.unk_token_id, 1, 0))
        col_index = torch.logical_and(col_index, real_tokens)
        num_tokens = torch.sum(col_index == True).int()
        if num_tokens < self.MAX_COL_TOKENS:
            labels[col_index] = inputs[col_index]
            inputs[col_index] = self.MASK_ID
        elif num_tokens < 10:
            # if a column is made of more than 5 tokens don't mask all, sample 50% for masking
            masked_indices = torch.logical_and(torch.bernoulli(inputs, self.COL_TOKEN_PROBABILITY).bool(), col_index)
            labels[masked_indices] = inputs[masked_indices] # Loss is calculated only on masked tokens
            inputs[masked_indices] = self.MASK_ID
        else:
            masked_indices = torch.logical_and(torch.bernoulli(inputs, self.MLM_PROBABILITY).bool(), col_index)
            labels[masked_indices] = inputs[masked_indices]  # Loss is calculated only on masked tokens
            inputs[masked_indices] = self.MASK_ID
        return inputs, labels

    def create_mask(self, data, col_encoding, col_len, col_index=0):
        # if the number of columns is 2 or less, simply mask the description and return 1 example
        # if the data is fred where every table has the same set of columns do the same thing - i.e.
        # mask descriptions and return one example
        if col_len <= 2:
            return self.create_desc_mask(data, col_encoding)

        masked_inputs, labels = self.create_desc_mask(data, col_encoding)
        # col index is 0 based, but we have column positions being encoded as 0 for SEP/CLASS, 1 for table desc
        col_index += 2
        self.mask_col(col_index, data, masked_inputs, labels)
        return masked_inputs, labels

    def create_desc_mask(self, data, col_encoding):
        # first column is description of table
        inputs = torch.from_numpy(numpy.array(data['input_ids']))
        real_tokens = numpy.where(inputs != self.tokenizer.unk_token_id, 1, 0)
        is_desc = torch.from_numpy(numpy.logical_and(numpy.where(col_encoding == 1, 1, 0), real_tokens))
        labels = inputs.clone()
        masked_indices = torch.logical_and(torch.bernoulli(inputs, self.MLM_PROBABILITY).bool(), is_desc)

        not_masked = torch.logical_not(masked_indices)
        labels[not_masked] = -100  # Loss is calculated only on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(inputs, 0.8).bool() & masked_indices
        inputs[indices_replaced] = self.MASK_ID

        # The rest of the time we replace masked tokens with random words
        indices_random = torch.logical_and(masked_indices, torch.logical_not(indices_replaced))
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        special_tokens = numpy.array([self.SEPARATOR_ID, self.CLS_ID, self.tokenizer.pad_token_id, self.tokenizer.unk_token_id])

        def condition(x):
            if x not in special_tokens:
                return True
            return False

        func = numpy.vectorize(condition)
        indices_real_tokens = torch.from_numpy(func(random_words))
        indices_random = torch.logical_and(indices_random, indices_real_tokens)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def save_tokenizer_checkpoint(self, chkpt_path):
        self.tokenizer.save_pretrained(chkpt_path)

    def create_col_value_encoding(self, col_pos_encoding, col_info, content_snapshot):
        l = []
        types = []
        token_pos = []
        min_hash_vals = []
        cnt = 0
        for i in col_pos_encoding:
            if i == 0:
                cnt = 0
            else:
                cnt += 1
            if i == 0:
                token_pos.append(0)
                #Zeros for CLS and SEP tokens
                arr = numpy.hstack([numpy.zeros(self.COUNT_UNIQUE), numpy.zeros(self.COUNT_NAN),
                                        numpy.zeros(self.CELL_WIDTH_BYTES),
                                        numpy.zeros(self.QUANTILE_LENGTH),
                                        numpy.zeros(self.PAD_VALUE)])
                min_hash = numpy.zeros(self.HIDDEN_SIZE)

                l.append(arr)
                types.append(0)
                min_hash_vals.append(min_hash)
            else:
                token_pos.append(cnt)
                # add table content snapshot
                if i == 1:
                    arr = numpy.zeros(self.HIDDEN_SIZE)
                    min_hash = numpy.hstack([numpy.array(content_snapshot)/ 1.0e8, numpy.zeros(self.HIDDEN_SIZE - self.MINHASH_LENGTH)])
                    l.append(arr)
                    types.append(0)
                    min_hash_vals.append(min_hash)
                    continue

                c = col_info[i - 2]
                if c['type'] not in ['string', 'date', 'integer', 'float']:
                    l.append(numpy.zeros(self.HIDDEN_SIZE))
                    types.append(0)
                    min_hash = numpy.zeros(self.HIDDEN_SIZE)
                    min_hash_vals.append(min_hash)
                    continue

                if 'unique' in c:
                    unique = numpy.array(c['unique'])
                else:
                    unique = numpy.zeros(self.COUNT_UNIQUE)

                if 'num_nan' in c:
                    num_nan =  numpy.array(c['num_nan'])
                else:
                    num_nan = numpy.zeros(self.COUNT_NAN)
                
                if 'cell_width_bytes' in c:
                    cell_width_bytes = c['cell_width_bytes']
                else:
                    cell_width_bytes = self.CELL_WIDTH_BYTES

                if 'min-hash-exact' in c:
                    min_hash_exact = numpy.array(c['min-hash-exact'])
                else:
                    min_hash_exact = numpy.zeros(self.MINHASH_LENGTH)

                if 'min-hash-words' in c:
                    min_hash_words = numpy.array(c['min-hash-words'])
                else:
                    min_hash_words = numpy.zeros(self.MINHASH_LENGTH)

                if 'quantile' in c:
                    quantile = numpy.array([0 if i is None or math.isnan(i) else i for i in c['quantile']])
                else:
                    quantile = numpy.zeros(self.QUANTILE_LENGTH)

                types.append(self.type_dict[c['type']])
                arr = numpy.hstack([unique, num_nan,
                                    cell_width_bytes,
                                    quantile,
                                    numpy.zeros(self.PAD_VALUE)])
                min_hash = numpy.hstack([min_hash_exact,  min_hash_words, numpy.zeros(self.MINHASH_PAD_VALUE)])

                assert len(arr) == self.HIDDEN_SIZE, c['type']
                assert not numpy.any(numpy.isnan(arr)), print(numpy.argwhere(numpy.isnan(arr)))

                if numpy.isfinite(arr).all():
                    arr = stats.zscore(arr)
                else:
                    arr = numpy.zeros(len(arr))
                min_hash = min_hash / 1.0e8
                l.append(arr)
                min_hash_vals.append(min_hash)
                assert len(arr) == self.HIDDEN_SIZE, c['type'] + ' ' + str(len(arr)) + ' ' + str(len(c['quantile']))
        assert len(l) == len(col_pos_encoding), str(len(l)) + " is not " + str(len(col_pos_encoding))
        assert len(types) == len(col_pos_encoding), str(len(types)) + " is not " + str(len(col_pos_encoding))
        return l, min_hash_vals, types, token_pos

    def tokenize_function(self, example, column_id=0):
        cols, str_cols = self.table_metadata_func(example, self)

        data = self.tokenizer.encode_plus(str_cols, max_length=self.config.max_position_embeddings, padding='max_length', truncation=True)#TODO: fix as param
        t = numpy.array(data['input_ids'])
        col_info = list(example['columns'].values())

        col_pos_encoding = self.create_col_encodings(t)
        data['position_ids'] = torch.tensor(col_pos_encoding)
        values, min_hash_vals, types, token_pos = self.create_col_value_encoding(col_pos_encoding, col_info, example['content_snapshot'])
        
        data['value_ids'] = torch.tensor(numpy.stack(values), dtype=torch.float)
        data['minhash_vals'] = torch.tensor(numpy.stack(min_hash_vals), dtype=torch.float)
        assert not data['value_ids'].isnan().any(), print(data['value_ids'])
        assert not data['minhash_vals'].isnan().any()
        data['token_type_ids'] = torch.tensor(types)
        data['token_position_ids'] = torch.tensor(token_pos)

        # add labels and masks
        # let us assume we always mask out a some portion of text description
        if not self.embedding_extraction:
            inputs, labels = self.create_mask(data, col_pos_encoding, len(cols), col_index=column_id)
        else:
            inputs, labels = torch.tensor(numpy.array(data['input_ids'])),torch.tensor(numpy.array(data['input_ids']))
        ret = {}
        for k, v in data.items():
            if isinstance(v, list):
                ret[k] = torch.tensor(numpy.array(v))
            else:
                ret[k] = v
        ret['input_ids'] = inputs
        return ret, labels
    
    
    
class TableSimilarityTokenizer(Tokenizer):

    def __int__(self, tokenizer, config, table_metadata_func=get_table_metadata_open_data):
        super.__init__(tokenizer, config, table_metadata_func)

    """
        This class subclasses the language modeling tokenizer, so column id - which is the 
        id to mask is ignored and set to a default value
    """
    def tokenize_function(self, example, column_id=0):
        # for table similarity, the test for positive and negative examples is that they share the same column set
        # however the order of the columns might vary across tables which will mean that the 'sentences' fed to the
        # model will be different.  So ensure same order.
        col_info = sorted(example['columns'].values(), key=lambda x: x['name'])
        cols = [c['name'] for c in col_info]
        # do not pass any metadata about table descriptions
        str_cols = 'table name' + self.SEPARATOR + self.SEPARATOR.join(cols)

        data = self.tokenizer.encode_plus(str_cols, max_length=self.config.max_position_embeddings,
                                        padding='max_length', truncation=True)
        t = numpy.array(data['input_ids'])

        col_pos_encoding = self.create_col_encodings(t)
        data['position_ids'] = torch.tensor(col_pos_encoding)
        values, min_hash_vals, types, token_pos = self.create_col_value_encoding(col_pos_encoding, col_info, example['content_snapshot'])
        data['value_ids'] = torch.tensor(numpy.stack(values), dtype=torch.float)
        data['minhash_vals'] = torch.tensor(numpy.stack(min_hash_vals), dtype=torch.float)
        # print(data['value_ids'].isnan().any())
        data['token_type_ids'] = torch.tensor(types)
        data['token_position_ids'] = torch.tensor(token_pos)


        ret = {}
        for k, v in data.items():
            if isinstance(v, list):
                ret[k] = torch.tensor(numpy.array(v))
            else:
                ret[k] = v

        return ret
