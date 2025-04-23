import pickle
import random
from argparse import ArgumentParser
import pytorch_lightning as pl
from tabsketchfm.data_processing.tabular_tokenizer import Tokenizer, fake_tablename_metadata
from transformers import AutoConfig, AutoTokenizer, BertConfig, BertTokenizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch, numpy as np, random
from tqdm import tqdm
# CCC Specific Code --------------------------------------------------------------------------------------------------------------------------
import os
import subprocess
import scipy

def find_table_col(toks, seq_states, table_start, table_end, inputs):
    cls_embedding = seq_states[0].cpu().tolist()
    special_tokens = set()
    special_tokens.add(toks.cls_token)
    special_tokens.add(toks.sep_token)
    special_tokens.add(toks.pad_token)

    tokens = toks.convert_ids_to_tokens(inputs)
    table_embeds = []
    #print(tokens)
    
    mask = []
    num_sep = 0

    col_states = {}
    #print('table start', table_start)
    #print('table end', table_end)
    
    for i in range(table_start, table_end):
        if tokens[i] in special_tokens:
            mask.append(False)
            if tokens[i] == toks.sep_token and i != table_start:
                #print('sep - so incrementing new col @', i)
                num_sep += 1
            continue
        else:
            mask.append(True)
            if num_sep not in col_states:
                col_states[num_sep] = []
            #print('adding token', tokens[i], i, 'to column num', num_sep)
            col_states[num_sep].append(seq_states[i])
    #print('num col_states', len(col_states))
    
    mask = torch.tensor(mask).unsqueeze(-1)

    seq_states = seq_states[table_start:table_end]

    sz = seq_states.size()[1]

    seq_states = seq_states.masked_select(mask)
    seq_states = torch.reshape(seq_states, (-1, sz))

    #sz = seq_states.size()
    #print('size', sz)
    #print('len of col states', len(col_states.values()))

    #print('seq_states size after mask', seq_states.size())    
    table_embedding = torch.mean(seq_states, dim=0)

    col_embeddings = {}
    for i in col_states:
        #print(col_states[i])
        t = torch.stack(col_states[i], dim=0)
        col_embeddings[i] = torch.mean(t, dim=0).cpu().tolist()
    table_embedding = table_embedding.cpu().tolist()
    
    
    return table_embedding, col_embeddings, cls_embedding
        

def ce_layer_by_layer_analysis(iterator, lmmodel, toks, l_itself, all_tables, batch_size):
    embeddings = {}

    with torch.no_grad():
        
        for idx, features in tqdm(enumerate(iterator)):
            # print('done in ce', idx, len(features[0]), features[1])
            input_features = features[0]
            input_len = features[0]['input_ids'].size()[1]
            num_t1_cols = features[3].cpu().tolist()
            table1_start = features[5].cpu().tolist()
            table1_end = features[6].cpu().tolist()
            table_name = features[2].cpu().tolist()

            model_predictions = lmmodel.model(**input_features, return_dict=True, output_hidden_states=True)
    
            # unclear why this model outputs data this way but the hidden states from the model is:
            # class transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
            # and despite the documentation, this is a list of tuples with the
            # first tuple of batch_size, seq_len, dimension is that of the embedding layer
            # and the rest 12 are from the layers
            
            hidden_states = model_predictions.hidden_states.hidden_states[1:]
            # any feature length will do here - batch size isnt the same for the last set
            local_batch_size = len(table1_start)
            #print('batch_size', batch_size)
            
            for layer in range(0, 1):
                layer_hidden_state = hidden_states[layer]
                table_set = set()
                for i in tqdm(range(0, local_batch_size)):
                    table_idx = (idx * batch_size) + i
                    table1_embedding, column_embeddings, cls_embedding = find_table_col(toks, layer_hidden_state[i], table1_start[i], table1_end[i], input_features['input_ids'][i])
                    if layer not in embeddings:
                        embeddings[layer] = []
                    assert all_tables[table_idx] == l_itself[table_idx]['table1']['filename']
                    assert all_tables[table_idx] not in table_set
                    table_set.add(all_tables[table_idx])

                    embeddings[layer].append({'table embedding': table1_embedding, 'column_embedding': column_embeddings, 'cls_embedding': cls_embedding, 'table':all_tables[table_idx]})

        return embeddings

def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="google-bert/bert-base-uncased")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--ground_truth', type=str)
    parser.add_argument('--task_type', type=str, default='regression')
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--outfile', type=str)
    
    args = parser.parse_args()


    from tabsketchfm import TableSimilarityTokenizer,FinetuneDataModule, FinetuneTabSketchFM 
       
    parser = FinetuneTabSketchFM.add_model_specific_args(parser)

    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.task_specific_params = {'hash_input_size': config.hidden_size}
    toks = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer = TableSimilarityTokenizer(tokenizer=toks, config=config, table_metadata_func=fake_tablename_metadata)

    all_tables = []

    import bz2
    import json
    for f in os.listdir(args.data_dir):
        if not f.endswith('.bz2'):
            continue
        with bz2.open(os.path.join(args.data_dir, f)) as inp:
            # print('reading', f)
            d = json.load(inp)
            all_tables.append(d['table_metadata']['file_name'])

    print('alltables len', len(all_tables))
    all_tables = list(set(all_tables))
    print('uniq all tables len', len(all_tables))

                
    l_itself = []
    for obj in all_tables:
        if ':' in obj:
            table = obj.split(':')[0] + '.csv'
            col = obj.split(':')[1]
            o = {}
            o['table1']={'filename':table,'col1':col}
            o['table2']={'filename':table,'col1':col}
            o['label']=1
            l_itself.append(o)
        else:
            o = {}
            o['table1']={'filename':obj}
            o['table2']={'filename':obj}
            o['label']=1
            l_itself.append(o)
    
    lmmodel = torch.load(args.checkpoint) 
    

    data_module = FinetuneDataModule(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            dataset={'test':l_itself},
            shuffle=False,
            concat = True,
            extract_embedding=True
        )

    batch_size = data_module.train_batch_size
    data_module.setup(None)
    iterator = data_module.test_dataloader()
    embeddings = ce_layer_by_layer_analysis(iterator, lmmodel, toks, l_itself, all_tables, batch_size)
    
    with open(args.outfile, 'wb') as out:
        pickle.dump(embeddings, out)

if __name__ == '__main__':

    cli_main()