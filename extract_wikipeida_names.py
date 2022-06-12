import logging, re

from pathlib import Path

from wikipedia_dump_db import DumpDB
from utils import check_path

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S',format='[%(filename)s:%(lineno)d] %(message)s')


class WikipediaPageNames():


    def __init__(self, path_db, data_dir):
        path_db = check_path(path_db)
        self.dir = check_path(data_dir, type='dir')
        self.db = DumpDB(str(path_db))

        self.russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.russian_alphabet = set([c for c in self.russian_chars])
        self.spec_symb = re.compile('\W+')
        self.numbers = re.compile('\d+')


    def build_list_wikipedia_names(self, compound_titles=True,
                                         max_num_words=3,
                                         upper_b_lenght=70,
                                         path_out=None,
                                         special_symb=False,
                                         russian_names=True,
                                         removed_numbers=True,
                                         limit=None):
        if limit is None:
            limit = int(1e9)
        if path_out is None:
            if compound_titles:
                suffix = 'comp'
            else:
                suffix = 'words'
            path_out = self.dir / f'wikipages_{suffix}.txt'

        with Path(path_out).open('w') as writer:
            for idx, title in enumerate(self.db.wiki_title_generator()):
                if not compound_titles:
                    if title.find(' ') > -1:
                        continue
                if len(title.split(' ')) > max_num_words:
                    continue
                if len(title) > upper_b_lenght:
                    continue
                if removed_numbers:
                    if self._is_numbers(title):
                        continue
                # remove with special symbols (stay only ,)
                if not special_symb:
                    if self._is_special_symbols(title):
                        continue
                if russian_names:
                    if not self._is_possible_ru_per_name(title):
                        continue


                line = f'{title}\n'
                writer.write(line)

                if idx % 1000 == 0:
                    logging.info(f'Extracted {idx} titles from wikipedia.')
                if idx > limit:
                    logging.info(f'Stop Iterations! Limit is exceed!')
                    break


    def _is_numbers(self, title):
        return len(re.sub(self.numbers, '', title)) < len(title)


    def _is_possible_ru_per_name(self, title):
        # if len(title.split(' ')) != 3:
        #     return False
        # if title.find(',') == -1:
        #     return False
        # if not all([w.istitle() for w in title.split(' ')]):
        #     return False
        if not self.is_cyrilyc(title, spec_symbs=(',', '(', ')', '"')):
            return False

        return True


    def is_cyrilyc(self, s, spec_symbs:tuple=(',')):
        s_copy = ''.join([c for c in s if c not in spec_symbs])
        for c in s_copy.lower():
            if c not in self.russian_alphabet:
                return False

        return True


    def _is_special_symbols(self, title, allowed_symbs=',)("[]'):
        # remove allowed symbols
        _title = title
        for c in allowed_symbs:
            _title = _title.replace(c, '')

        return len(re.sub(self.spec_symb, '', _title)) < len(_title)


def build_wikinames_list(data_dir, path_db, compound_titles=True, max_num_words=3,
                         upper_b_lenght=50, russian_names=True, limit=None):
    wikipages = WikipediaPageNames(path_db, data_dir=data_dir)
    wikipages.build_list_wikipedia_names(compound_titles,
                                         max_num_words,
                                         upper_b_lenght,
                                         russian_names=russian_names,
                                         limit=limit)


dir = '/home/mklochkov/projects_python/CNNentityEmbedd/data'
path_db = '/home/mklochkov/projects_python/data/dictionaries/output_db_all.db'


if __name__ == '__main__':
    build_wikinames_list(data_dir=dir, path_db=path_db)