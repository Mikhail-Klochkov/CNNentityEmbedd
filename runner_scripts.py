
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import check_path
from string_dataset_builder import StringDatasetBuilder, StringDatasetOneHot
from networks import CNNString

path_eng_words_article = '/home/mklochkov/projects_python/CNNentityEmbedd/data/paper_words.txt'
path_wiki_entities_sing = '/home/mklochkov/projects_python/CNNentityEmbedd/data/wikipages_no_comp.txt'


class RunnerScripts():

    @staticmethod
    def build_one_hot_string_dataset(path_words,
                                     train_ratio=0.7,
                                     upper_bound_len=50,
                                     limit=None):
        path_words = check_path(path_words)
        string_dataset = StringDatasetBuilder(path_words,
                                              'one_hot',
                                              train_ratio,
                                              upper_bound_len,
                                              limit)
        alphabet = string_dataset.alphabet
        max_lenght = string_dataset.max_lenght
        one_hot_encoders = string_dataset.one_hot

        return one_hot_encoders, alphabet, max_lenght


    @staticmethod
    def run_dataloader_string(path_words, train_ratio=0.7, upper_bound_len=50, batch_size=128,
                              limit=None):
        one_hot, alphabet, max_lenght = RunnerScripts.build_one_hot_string_dataset(path_words,
                                                                               train_ratio,
                                                                               upper_bound_len,
                                                                               limit)
        dataset = StringDatasetOneHot(one_hot, max_lenght, alphabet, return_string=True)
        datalader = DataLoader(dataset, batch_size, shuffle=True)
        for one_hot_embs, strings in tqdm(datalader):
            print(one_hot_embs.shape)


if __name__ == '__main__':
    #RunnerScripts.build_one_hot_string_dataset(path_eng_words_article, limit=10000)
    RunnerScripts.run_dataloader_string(path_eng_words_article)
