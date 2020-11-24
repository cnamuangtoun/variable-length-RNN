import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import make_x_y
from .metrics import accuracy
from .model import Model



def train():
    #Init data.
    torch.manual_seed(0)

    BATCH_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 250

    train_dataset, test_dataset = text_classification \
        .DATASETS['YelpReviewPolarity'](root='./.data',
        ngrams=2, vocab=None)

    sub_train_, sub_valid_ = \
        random_split(train_dataset, [SEQ_LEN, len(train_dataset) - SEQ_LEN])

    dataloader_train = DataLoader(sub_train_, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=make_x_y)

    dataloader_dev = DataLoader(sub_valid_, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=make_x_y)

    #Init model.
    model = Model().to(device)
    loss_fn = nn.NLLLoss(weight=classweights, reduction='mean')
    loss_fn_test = nn.NLLLoss(reduction='mean')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    #Initial test.
    model.eval()
    losses, acc_arr = [], []
    with torch.no_grad():
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            losses.append(loss_fn_test(y_pred, y).cpu())
            acc_arr.append(accuracy(y_pred, y).cpu())
    loss_init = torch.tensor(losses).mean().item()
    t1_init = torch.tensor(acc_arr).mean().item()
    print(f'E0 TEST:{loss_init:.3f}')

    #Training.
    loss_min = float('inf')
    max_unimproved_epochs, unimproved_epochs = 15, 0
    for epoch in range(1, 1000):
        start_time = time.time()
        #Training.
        model.train()
        losses = []
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            assert torch.isfinite(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.cpu())
        loss_train = torch.tensor(losses).mean().item()
        #Testing.
        model.eval()
        losses, acc_arr = [], []
        with torch.no_grad():
            for X, y in dataloader_dev:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn_test(y_pred, y)
                assert torch.isfinite(loss)
                losses.append(loss.cpu())
                acc_arr.append(accuracy(y_pred, y).cpu())
        loss_test = torch.tensor(losses).mean().item()
        t1_test = torch.tensor(acc_arr).mean().item()
        #Feedback.
        print(f'E{epoch} TRAIN:{loss_train:.3f} '
            f'TEST:{loss_test:.3f} ACC:{t1_test:.3f} '
            f'TOOK:{time.time() - start_time:.2f}')
        #Save state & early stopping.
        unimproved_epochs += 1
        if loss_test < loss_min:
            loss_min, t1 = loss_test, t1_test
            torch.save(model.state_dict(), FILEPATHS['model'])
            unimproved_epochs = 0
        if unimproved_epochs > max_unimproved_epochs:
            print(f'E{epoch} Early stopping. BEST TEST:{loss_min:.3f}')
            break

    return (loss_init, loss_min), (t1_init, t1)
