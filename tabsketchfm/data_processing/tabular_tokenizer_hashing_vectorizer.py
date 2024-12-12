from .tabular_tokenizer import Tokenizer, get_table_metadata_open_data
import numpy
from sklearn.feature_extraction.text import HashingVectorizer


class Tokenizer_HV(Tokenizer):

    def __init__(self, tokenizer, config, table_metadata_func=get_table_metadata_open_data, embedding_extraction=False):
        super().__init__(tokenizer, config, table_metadata_func, embedding_extraction)
        self.HV_SIZE = HashingVectorizer().n_features

    def create_col_value_encoding(self, col_pos_encoding, col_info, content_snapshot):
        types = []
        token_pos = []
        hv_vals = []
        l = []
        cnt = 0
        for i in col_pos_encoding:
            if i == 0:
                cnt = 0
            else:
                cnt += 1
            if i == 0:
                token_pos.append(0)
                hv = numpy.zeros(self.HV_SIZE)
                l.append(numpy.zeros(self.HIDDEN_SIZE))
                types.append(0)
                hv_vals.append(hv)
            else:
                token_pos.append(cnt)
                # add table content snapshot
                if i == 1:
                    types.append(0)
                    l.append(numpy.zeros(self.HIDDEN_SIZE))
                    hv = numpy.zeros(self.HV_SIZE)
                    hv_vals.append(hv)
                    continue

                c = col_info[i - 2]
                if c['type'] not in ['string', 'date', 'integer', 'float']:
                    types.append(0)
                    l.append(numpy.zeros(self.HIDDEN_SIZE))
                    hv = numpy.zeros(self.HV_SIZE)
                    hv_vals.append(hv)
                    continue
                l.append(numpy.zeros(self.HIDDEN_SIZE))
                types.append(self.type_dict[c['type']])
                if 'hv' in c:
                    hv = numpy.array(c['hv'])
                else:
                    hv = numpy.zeros(self.HV_SIZE)
                hv_vals.append(hv)
        assert len(types) == len(col_pos_encoding), str(len(types)) + " is not " + str(len(col_pos_encoding))
        return l, hv_vals, types, token_pos


class TableSimilarityTokenizer_HV(Tokenizer_HV):

    def __int__(self, tokenizer, config, table_metadata_func=get_table_metadata_open_data):
        super.__init__(tokenizer, config, table_metadata_func)
