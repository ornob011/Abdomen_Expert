from get_model_optimizer import get_model, get_optimizer
from dataloader import get_dataloader
from sklearn.model_selection import KFold
import torch
from tqdm.notebook import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import gc
import time


k_folds = 10
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 2
DEVICE = 'cuda'


kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

criterion = torch.nn.MarginRankingLoss(1.0)

optimizer = get_optimizer()
model = get_model()
trainset = get_dataloader()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=0, verbose=True)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train_fn(epoch, n_epochs, model, dataloader, optimizer, criterion):

    model.train()
    torch.set_grad_enabled(True)
    epoch_loss = 0.0
    accuracies = [0, 0]
    acc_threshes = [0.5, 0.7]

    tq_batch = tqdm(dataloader, total=len(dataloader))
    for A, P, N, label in tq_batch:

        A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

        A, P, N = Variable(A), Variable(P), Variable(N)

        A_ends = model(A)
        P_ends = model(P)
        N_ends = model(N)

        dist_E1_E2 = F.pairwise_distance(A_ends, P_ends, 2)
        dist_E1_E3 = F.pairwise_distance(A_ends, N_ends, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(DEVICE)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        tq_batch.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        tq_batch.set_postfix_str('loss = {:.4f}'.format(loss.item()))

        del A
        del P
        del N
        gc.collect()

        for i in range(len(accuracies)):
            prediction = (dist_E1_E3 - dist_E1_E2 - 1.0 *
                          acc_threshes[i]).cpu().data
            prediction = prediction.view(prediction.numel())
            prediction = (prediction > 0).float()
            batch_acc = prediction.sum() * 1.0 / prediction.numel()
            accuracies[i] += batch_acc

        torch.cuda.empty_cache()

    for i in range(len(accuracies)):
        print('Train Accuracy with diff = {}% of margin: {}'.format(
            acc_threshes[i] * 100, accuracies[i] / len(dataloader)))
        torch.cuda.empty_cache()

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = accuracies[0] / len(dataloader)

    return epoch_loss, epoch_acc


def eval_fn(model, dataloader, criterion):

    model.eval()

    epoch_loss = 0.0
    accuracies = [0, 0]
    acc_threshes = [0.5, 0.7]

    tq_batch = tqdm(dataloader, total=len(dataloader), leave=False)

    with torch.no_grad():
        for A, P, N, label in tq_batch:

            A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

            A, P, N = Variable(A), Variable(P), Variable(N)

            E1 = model(A)
            E2 = model(P)
            E3 = model(N)

            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)
            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(DEVICE)
            target = Variable(target)
            loss = criterion(dist_E1_E2, dist_E1_E3, target)

            epoch_loss += loss.item()

            del A
            del P
            del N
            gc.collect()

            for i in range(len(accuracies)):
                prediction = (dist_E1_E3 - dist_E1_E2 - 1.0 *
                              acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc

        for i in range(len(accuracies)):
            print('Test Accuracy with diff = {}% of margin: {}'.format(
                acc_threshes[i] * 100, accuracies[i] / len(dataloader)))
            torch.cuda.empty_cache()

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = accuracies[1] / len(dataloader)
    return epoch_loss, epoch_acc


def fit(n_epochs, model, optimizer):

    since = time.time()

    history = []
    foldperf = {}
    best_valid_loss = float('inf')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):

        print(f'FOLD {fold+1}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                  batch_size=BATCH_SIZE, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                 batch_size=BATCH_SIZE, sampler=test_subsampler)

        print(f"Number of batches in trainloader : {len(trainloader)}")
        print(f"Number of batches in validloader : {len(testloader)}")

#         model.apply(reset_weights)
        history2 = {'train_loss': [], 'val_loss': [],
                    'train_acc': [], 'val_acc': []}

        for epoch in range(n_epochs):

            torch.cuda.empty_cache()

            train_loss, train_acc = train_fn(
                epoch, n_epochs, model, trainloader, optimizer, criterion)
            valid_loss, valid_acc = eval_fn(model, testloader, criterion)

            scheduler.step(valid_loss)

            history.append({'train_loss': train_loss, 'train_acc': train_acc,
                           'val_loss': valid_loss, 'val_acc': valid_acc})

            statement = "[loss]={:.4f} - [val_loss]={:.4f}".format(
                train_loss,  valid_loss)
#             print(statement)
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.4f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                                     epoch+1,
                                                                                                                                     train_loss,
                                                                                                                                     valid_loss,
                                                                                                                                     train_acc,
                                                                                                                                     valid_acc))
            history2['train_loss'].append(train_loss)
            history2['val_loss'].append(valid_loss)
            history2['train_acc'].append(train_acc)
            history2['val_acc'].append(valid_acc)

            if valid_loss < best_valid_loss:

                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion}, 'proposed_model_all.pt')

                best_valid_loss = valid_loss
                print("SAVED_WEIGHTS_SUCCESS")

        foldperf['fold{}'.format(fold+1)] = history2

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return history, foldperf


def train_cnn():

    res, foldperf = fit(EPOCHS, model, optimizer)
    return res, foldperf
