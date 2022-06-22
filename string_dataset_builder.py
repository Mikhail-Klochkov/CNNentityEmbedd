import logging, numpy as np, torch

from torch.utils.data import Dataset
from typing import Dict, Union

from utils import check_path


path_names = '/home/mklochkov/projects_python/CNNentityEmbedd/data/wikipages_no_comp.txt'


class Alphabet():


    def __init__(self, id_to_char: Dict[int, str], char_to_id: Dict[str, int]):
        assert len(id_to_char) == len(char_to_id), 'Incorrect behaviour'
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


    def from_ids_to_string(self, ids):
        return ''.join([self.get_c(id) for id in ids])

    # not compound words need decode each
    def from_one_hot_to_string(self, one_hot: Union[np.ndarray, torch.Tensor],
                                     compound:bool=False,
                                     string_check:str=None):
        if torch.is_tensor(one_hot):
            if one_hot.requires_grad:
                one_hot = one_hot.detach().cpu().numpy()
            else:
                one_hot = one_hot.cpu().numpy()

        chars = []
        for pos_char in range(one_hot.shape[1]):
            mask = np.where(one_hot[:, pos_char] == 1.)[0]
            if len(mask) == 0:
                if not compound:
                    string = ''.join(chars)
                    if string_check:
                        assert string_check == string, f'not equal : {string} != {string_check}'
                    return string
                else:
                    continue
            assert len(mask) == 1, 'Incorrect behaviour. Should be one symbol!'
            chars += [self.id_to_char[mask[0]]]

        return ''.join(chars)


    def get_id(self, c:str):
        if c not in self.char_to_id:
            return None
        return self.char_to_id[c]


    def __len__(self):
        return len(self.id_to_char)


    def get_c(self, id:int):
        if id not in self.id_to_char:
            return None
        return self.id_to_char[id]


class StringDatasetRuBert(Dataset):

    def __init__(self):
        pass


class StringDatasetOneHot(Dataset):


    def __init__(self, encodings, max_lenght, alphabet, return_string=False):
        # this is list of lists
        self.encodings = encodings
        # max lenght of string
        self.max_lenght = max_lenght
        # should be class user object
        # from id -> char, char -> id
        self.alphabet = alphabet
        # specific parameters
        self.return_strings = return_string


    # need extract one-hot-embedding or RuBert embedding representation
    def __getitem__(self, idx):
        # list if indeces (id <-> char)
        # for each string is different lenght
        string_repr = self.encodings[idx]
        one_hot_emb = np.zeros((len(self.alphabet), self.max_lenght), dtype=np.float32)
        one_hot_emb[string_repr, np.arange(len(string_repr))] = 1.
        if self.return_strings:
            # with string
            string = ''.join([self.alphabet.get_c(id_char) for id_char in string_repr])
            return one_hot_emb, string
        else:
            return one_hot_emb


    def __len__(self):
        return len(self.encodings)


class StringDatasetBuilder():


    def __init__(self, path, type_embedding='one_hot',
                             train_ratio=0.7,
                             upper_bound_max_len=70,
                             limit=None):

        self.path = check_path(path)
        # this we build
        if type_embedding == 'one_hot':
            self.alphabet, self.one_hot, self.max_lenght = self.build_one_hot_strings(
                                                        train_ratio, upper_bound_max_len, limit)
        elif train_ratio == 'RuBERT':
            self.build_bert_repr_strings()
        else:
            raise ValueError(f'Incorrect type_embedding: {type_embedding}!')


    def build_one_hot_strings(self, train_ratio=0.7, upper_bound_max_len=70, limit=None):

        if limit is None:
            limit = int(1e9)

        if train_ratio > 1. or train_ratio < 0:
            raise ValueError(f'Incorrect train_ratio: {train_ratio} parameter.')
        alpha_char_to_id = {}
        alpha_id_to_char = {}
        curr_char_id = 0
        max_len = 0
        encods = []
        with self.path.open() as reader:
            for idx_line, line in enumerate(reader):
                line = line.rstrip('\n').lower()
                # ignore very long strings
                if len(line) > upper_bound_max_len:
                    continue
                if limit < idx_line:
                    logging.info(f'Stop Iterations!')
                    break
                max_len = max(max_len, len(line))
                string_code = []
                for c in line:
                    if c not in alpha_char_to_id:
                        alpha_char_to_id[c] = curr_char_id
                        alpha_id_to_char[curr_char_id] = c
                        curr_char_id += 1

                    char_id = alpha_char_to_id[c]
                    string_code += [char_id]

                encods.append(string_code)

                if idx_line % 10000 == 0:
                    logging.info(f'Extracted {idx_line} data strings!')

        # need ignore some specific names
        if train_ratio:
            alpha_char_to_id_new = {}
            alpha_id_to_char_new = {}
            max_len_new = 0
            cur_char_id_new = 0

            indeces = np.arange(len(encods))
            np.random.shuffle(indeces)
            indeces = indeces[: int(train_ratio * len(encods))]
            encods = [encods[idx] for idx in indeces]
            encods_new = []
            for one_hot_per_str in encods:
                str_chars = [alpha_id_to_char[id] for id in one_hot_per_str]
                max_len_new = max(max_len_new, len(str_chars))
                str_code_new = []
                for char in str_chars:
                    if char not in alpha_char_to_id_new:
                        alpha_char_to_id_new[char] = cur_char_id_new
                        alpha_id_to_char_new[cur_char_id_new] = char
                        cur_char_id_new += 1

                    str_code_new += [alpha_char_to_id_new[char]]
                encods_new.append(str_code_new)

            # new encodings
            max_len = max_len_new
            alpha_char_to_id = alpha_char_to_id_new
            alpha_id_to_char = alpha_id_to_char_new
            encods = encods_new

        # Generate new object
        alphabet = Alphabet(id_to_char=alpha_id_to_char, char_to_id=alpha_char_to_id)

        return alphabet, encods, max_len



    def build_bert_repr_strings(self):
        raise NotImplementedError