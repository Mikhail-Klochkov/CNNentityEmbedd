import logging, numpy as np

from typing import Dict
from utils import check_path


path_names = '/home/mklochkov/projects_python/CNNentityEmbedd/data/wikipages_no_comp.txt'


class Alphabet():


    def __init__(self, id_to_char: Dict[int, str], char_to_id: Dict[str, int]):
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


    def get_id(self, c:str):
        if c not in self.char_to_id:
            return None
        return self.char_to_id[c]


    def get_c(self, id:int):
        if id not in self.id_to_char:
            return None
        return self.id_to_char[id]



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