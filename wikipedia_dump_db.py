import lmdb, logging, mwparserfromhell, multiprocessing, re, six, zlib
from uuid import uuid1
from functools import partial
from pathlib import Path
from contextlib import closing
from six.moves import cPickle as pickle
from multiprocessing.pool import Pool

logger = logging.getLogger(__name__)
STYLE_RE = re.compile("'''*")


class Paragraph:


    def __init__(self, text, wiki_links, abstract):
        self.text = text
        self.wiki_links = wiki_links
        self.abstract = abstract


    def __repr__(self):
        return '<Paragraph %s>' % (self.text[:50] + '...')


    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))


class WikiLink:


    def __init__(self, title, text, start, end):
        self.title = title
        self.text = text
        self.start = start
        self.end = end


    @property
    def span(self):
        return (self.start, self.end)


    def __repr__(self):
        return '<WikiLink %s->%s>' % (self.text, self.title)


    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.start, self.end))


class DumpDB:


    def  __init__(self, db_file):
        self.db_file = db_file
        self.env = lmdb.open(db_file, readonly=True, subdir=False, lock=False, max_dbs=3)
        self.meta_db = self.env.open_db(b'__meta__')
        self.page_db = self.env.open_db(b'__page__')
        self.redirect_db = self.env.open_db(b'__redirect__')


    def extract_all_titles_in_txt_file(self, output_file):
        if isinstance(output_file, str):
            output_file = Path(output_file)

        with self.env.begin(db = self.page_db) as txn, output_file.open('w') as writer:
            cur = txn.cursor()
            for idx, key in enumerate(cur.iternext(values=False)):
                title = key.decode('utf-8')
                writer.write(title + '\n')
                if idx % 50000:
                    print(f'Idx: {idx}.')


    def redirect_size(self):
        with self.env.begin(db=self.redirect_db) as txn:
            return txn.stat()['entries']


    def titles(self):
        with self.env.begin(db=self.page_db) as txn:
            cur = txn.cursor()
            for key in cur.iternext(values=False):
                yield key.decode('utf-8')


    def redirects(self):
        with self.env.begin(db=self.redirect_db) as txn:
            cur = txn.cursor()
            for (key, value) in iter(cur):
                yield (key.decode('utf-8'), value.decode('utf-8'))


    def get_redirect_title(self, title, with_checking_pages=True):
        with self.env.begin(db = self.redirect_db) as txn, self.env.begin(db = self.page_db) as txn_pages:
            value = txn.get(title.encode('utf-8'))
            if value:
                title_redirect = value.decode('utf-8')
                if with_checking_pages:
                    page_value = txn_pages.get(value)
                    if page_value:
                        return title_redirect, True
                    else:
                        return title_redirect, False
            else:
                return None


    def resolve_redirect(self, title):
        with self.env.begin(db=self.redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))
            if value:
                return value.decode('utf-8')
            else:
                return title


    def is_redirect(self, title):
        with self.env.begin(db=self.redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))

        return bool(value)


    def is_in_page_database(self, title):
        with self.env.begin(db=self.page_db) as txn:
            value = txn.get(title.encode('utf-8'), default=None)
            if value:
                return True
            else:
                return False


    def is_disambiguation(self, title):
        with self.env.begin(db=self.page_db) as txn:
            value = txn.get(title.encode('utf-8'))

        if value is None:
            raise KeyError

        return pickle.loads(zlib.decompress(value))[1]


    def get_paragraphs(self, key):
        with self.env.begin(db=self.page_db) as txn:
            value = txn.get(key.encode('utf-8'))
            if not value:
                raise KeyError(key)

        return self.deserialize_paragraphs(value)


    def deserialize_paragraphs(self, value):
        ret = []
        for obj in pickle.loads(zlib.decompress(value))[0]:
            wiki_links = [WikiLink(*args) for args in obj[1]]
            ret.append(Paragraph(obj[0], wiki_links, obj[2]))

        return ret
