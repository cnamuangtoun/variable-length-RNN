import torch
import pandas as pd
import torch.nn as nn
from torchtext import datasets
from torchtext.data import Field, Iterator, TabularDataset


class Dataset(data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')

        self.TEXT = Field(lower=True, tokenize='spacy', batch_first = True)
		self.LABEL = Field(sequential=False, unk_token = None, is_target = True)

		self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

		self.TEXT.build_vocab(self.train, self.dev)
		self.LABEL.build_vocab(self.train)

		vector_cache_loc = '.vector_cache/snli_vectors.pt'
		if os.path.isfile(vector_cache_loc):
			self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
		else:
			self.TEXT.vocab.load_vectors('glove.840B.300d')
			makedirs(os.path.dirname(vector_cache_loc))
			torch.save(self.TEXT.vocab.vectors, vector_cache_loc)

		self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test),

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return (
            torch.tensor(self.X[ix], dtype=torch.float32,
                device=self.device, requires_grad=False),
            torch.tensor(self.y[ix], dtype=torch.int64,
                device=self.device, requires_grad=False)
        )



TEXT = Field(lower=True, tokenize='spacy', batch_first = True)
LABEL = Field(sequential=False, unk_token=None, is_target = True)

train, dev, test = datasets.SNLI.splits(TEXT, LABEL)
train[:5]
