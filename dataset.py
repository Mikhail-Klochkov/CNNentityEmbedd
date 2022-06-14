import spacy, logging

from utils import check_path

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S',format='[%(filename)s:%(lineno)d] %(message)s')


path_names = '~/projects_python/CNNentityEmbedd/data/wikipages_comp.txt'


class Dataset():


    def __init__(self, path):
        self.path = check_path(path)
        self.max_lenght, self.dictionary_symbols = self.read_dataset()


    def read_dataset(self, train_dataset=0.7):

        dictionary_symbols = set()
        max_lenght = 0

        with self.path.open() as reader:
            for idx_line, line in enumerate(reader):
                line = line.rstrip('\n')
                if len(line) > max_lenght:
                    max_lenght = len(line)
                for c in line:
                    dictionary_symbols.add(c)

                [dictionary_symbols.add(c) for c in line]
                if idx_line % 10000 == 0:
                    logging.info(f'Extracted {idx_line} data strings!')

        return max_lenght, dictionary_symbols


    def one_hot_encoder(self):
        pass