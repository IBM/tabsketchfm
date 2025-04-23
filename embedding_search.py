import pickle
import sys
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score
import statistics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
import pandas
from tqdm import tqdm
import os
from argparse import ArgumentParser

from sentence_transformers import SentenceTransformer

# sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')


def normalize(t):
   mean, std, var = np.mean(t), np.std(t), np.var(t) 
   t  = (t-mean)/std
   return t
   
def get_table_cols(t, data_dir, get_embeddings=False):
      
   df = pandas.read_csv(os.path.join(data_dir, t))
   value_embeddings = None
   if get_embeddings:
      value_embeddings = []
      for c in df.columns:
         
         l = df[c].unique().tolist()[:100]
         l = [str(x).strip() for x in l if str(x).strip() != '' and x is not None]
         sent = ','.join(l)
         x = sent_model.encode(sent)
         value_embeddings.append(normalize(x))
      assert len(df.columns) == len(value_embeddings)

   cols = [x.lower().strip() for x in df.columns]
   
   return len(df.columns), value_embeddings, cols

def check_table(t, embeddings):
   num_cols, _, _ = get_table_cols(t)
   for i in range(0, num_cols):
      key = t + ':' + str(i)
      assert key in embeddings, key

def get_table_embeddings(layer, layer_embeddings, test_set_size, data_dir, by_cols=True, union=True, add_values=None):
   
   table2embeddings = {}

   #print('length of embeddings', len(layer_embeddings[layer]))
   for obj in layer_embeddings[layer]:
      if by_cols:

         if ':' in obj['table']:
            arr = obj['table'].split(':')
            table = arr[0]
            col = arr[1]
            table2embeddings[table + ":" + col] = obj['column_embedding'][int(col)]
         else:
            if not union:
               table = os.path.basename(obj['table']).replace('.csv', '')
            else:
               table = os.path.basename(obj['table'])

            if add_values:
               _, value_embeddings,columns = get_table_cols(table, data_dir, get_embeddings=True)

            for ce in obj['column_embedding']:

               if add_values:
                  # Tabsketchfm can add extra columns for dates which are always last, so we can ignore
                  # extra columns
                  if ce >= len(value_embeddings):
                     print('missing column')
                     continue
                  # print(table, ce)
                  if add_values == 'concatenate':
                     table2embeddings[table + ":" + str(ce)] = np.concatenate((value_embeddings[ce], normalize(np.array(obj['column_embedding'][ce]))),axis=None)
                  elif add_values == 'values_only':
                     table2embeddings[table + ":" + str(ce)] = value_embeddings[ce]
                  else:
                     raise "Unknown parameter for add_values - options are values_only or concatenate"
               else:
                  table2embeddings[table + ":" + str(ce)] = obj['column_embedding'][ce]
      else:
         table2embeddings[obj['table']] = obj['table embedding']
   return table2embeddings

   
def handle_layer(layer, layer_embeddings, search2results, data_dir, outfile, test_set_size, by_cols=True, union=True, k=10, add_values=None):
   table_embeddings = get_table_embeddings(layer, layer_embeddings, test_set_size, data_dir, by_cols, union, add_values)
   print("Got all the embeddings for indexing")
   table_order = list(table_embeddings.keys())
   
   neigh = NearestNeighbors(metric='cosine')

   all_embed = []

   for _,v in table_embeddings.items():
      if not isinstance(v, np.ndarray):
         all_embed.append(np.array(v))
      else:
         all_embed.append(v)       
      
   all_embed = np.array(all_embed)

   neigh.fit(all_embed)
   print("Indexed")
   results = {}
   
   for idx, key in enumerate(tqdm(search2results)):
      expected = search2results[key]

      if union:
         num_cols, _, _ = get_table_cols(key, data_dir)

         
         table2qcols = {}
         
         for i in range(0, num_cols):
            t_key = key + ":" + str(i)
            if t_key not in table_embeddings:
               print('missing key', t_key)
               continue
            nbr_dist, nbrs = neigh.kneighbors([table_embeddings[t_key]], n_neighbors=k*3, return_distance=True)
            nbr_dist = nbr_dist[0]
            nbrs = nbrs[0]
            matched = [table_order[j] for j in nbrs]
            # filter match by first matched col for any returned table
            firstmatch = set()
            for dist_idx, m in enumerate(matched):
               arr = m.split(':')
               if arr[0] not in firstmatch:
                  firstmatch.add(arr[0])
                  if arr[0] not in table2qcols:
                     table2qcols[arr[0]] = []
                  table2qcols[arr[0]].append((arr[1], nbr_dist[dist_idx]))

         table2qcols_sums = {}
         for x in table2qcols:
            v = table2qcols[x]
            v = sorted(v, key=lambda item:item[1])
            seen_cols = set()
            l = []
            # ensure to add each column only once in the sum
            for c, d in v:
               if c in seen_cols:
                  continue
               else:
                  seen_cols.add(c)
                  l.append(d)
                  
            table2qcols_sums[x] = (len(l), sum(l))
            
         sorted_table2qcols = dict(sorted(table2qcols_sums.items(), key=lambda item: (-item[1][0], item[1][1])))
         
         nbrs = [x for x, y in sorted(table2qcols_sums.items(), key=lambda item: (-item[1][0], item[1][1]))][:k+1]
         
         expect = expected
         neighbors = nbrs

      else:
         
         if key not in table_embeddings:
            print('cant find key!', key)
            continue
         
         nbrs = neigh.kneighbors([table_embeddings[key]], n_neighbors=k+1, return_distance=False)[0]

         # print("key", key)         
         expect = []
         for val in search2results[key]:
            if val in table_order:
               expect.append(table_order.index(val))
         
         nbrs = nbrs.tolist()
         neighbors = [table_order[x] for x in nbrs]
      
         if key in nbrs:
            nbrs.remove(table_order.index(key))
               
      results[key] = neighbors
 
   
   with open(outfile, 'wb') as out:
      pickle.dump(results, out)


"""
for i in layer_embeddings:
   print('*********layer************', i)
   handle_layer(i, layer_embeddings, search2results, by_cols=True)
"""
def cli_main():

   # ------------
   # args
   # ------------
   parser = ArgumentParser()
   parser.add_argument('--embeddings', type=str)
   parser.add_argument('--ground_truth', type=str)
   parser.add_argument('--k', type=int, default=10)
   parser.add_argument('--use_column_based_table_search', type=str, default='False')
   parser.add_argument('--by_cols', type=str, default='True')
   parser.add_argument('--data_dir', type=str)
   parser.add_argument('--outfile', type=str)
   parser.add_argument('--add_values', type=str, default=None, help="Adding the Sentence Transformer Embeddings. Options are concatenate or values_only.")
    
   args = parser.parse_args()
   k = args.k
   if args.use_column_based_table_search =='True':
      union = True
   else:
      union = False 
   
   if args.by_cols =='True':
      by_cols = True
   else:
      by_cols = False

   print("loading embedding")
   with open(args.embeddings, 'rb') as f:
         layer_embeddings = pickle.load(f)
            
   print("loaded embedding")

   with open(args.ground_truth, 'rb') as f:
      search2results = pickle.load(f)
      test_set_size = len(search2results)
   print("Loaded ground truth")
   handle_layer(0, layer_embeddings, search2results, args.data_dir, args.outfile, test_set_size, by_cols=by_cols, union=union, k=args.k, add_values=args.add_values)

if __name__ == '__main__':

    cli_main()
