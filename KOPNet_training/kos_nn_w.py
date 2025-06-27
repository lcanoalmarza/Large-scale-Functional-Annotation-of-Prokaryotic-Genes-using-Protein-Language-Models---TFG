#!/usr/bin/env python

"""
Make a predictor of ko annotations using a neural network

"""

import sys
from sklearn.metrics import recall_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as fmt
from torch import nn, optim, cuda, no_grad, save
from dataloader_w import get_dataloaders


# get_args
def get_args():
    """Return the arguments passed in the command line."""
    parser = ArgumentParser(description=__doc__, formatter_class=fmt)

    add = parser.add_argument  # shortcut
    add('--samples', default='data/test_umap_result.npy',
        help='data file with a list of numbers characterizing each protein')
    add('--labels', default='data/test_ko_labels.csv',
        help='data file with the ko corresponding to each protein')
    add('-e', '--epochs', type=int, default=8,
        help='number of epochs (iterations over the full training data)')
    add('-n', '--hidden-sizes', type=int, nargs='+', default=[100],
        help='neurons in the hidden layers')
    add('-t', '--test-size', type=float, default=0.25,
        help='fraction of data reserved for testing (not used for training)')
    add('-s', '--shuffle', action='store_true',
        help='shuffle data before separating into training and testing data')
    add('-l', '--learning-rate', type=float, default=1e-3,
        help='learning rate (size of the step following the gradient)')
    add('-b', '--batch-size', type=int, default=64,
        help='batch size (number of samples to feed per model call)')
    add('-d', '--device', choices=['cpu', 'cuda', 'auto'], default='auto',
        help='device where to do the computations ("cuda" for GPU)')
    add('-o', '--output',
        help='file where to save the final model parameters')
    add('-q', '--quiet', action='store_true',
        help='be less verbose')

    return parser.parse_args()

# class
class ResidualLinearBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.relu = nn.LeakyReLU(0.1)

        self.proj = nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.linear(x)
        out = self.bn(out)
        out = out + residual
        return self.relu(out)

class ko_ProstT5_NN(nn.Module):
    """Neural network to predict KO from embeddings (e.g., UMAP or ProstT5)"""

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        size_previous = input_size
        for size in hidden_sizes:
            layers.append(ResidualLinearBlock(size_previous, size))
            size_previous = size

        self.linear_relu_stack = nn.Sequential(*layers)
        self.output_layer = nn.Linear(size_previous, output_size)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        out = self.output_layer(out)
        return out



# train
def train(dataloader, model, loss_fn, optimizer, quiet=False):
    """Train the model (update its parameters).

    The update is done by comparing the model predictions for inputs
    (from the dataloader) to the correct expected outputs (also coming
    from the dataloader). The comparison uses loss_fn, and the updates
    are done according to the optimizer's strategy.
    """
    model.train()  # set model in training mode

    for batch, (X, y) in enumerate(dataloader):  # dataloader for training data
        X, y = X.to(device), y.to(device)  # move to computing device

        optimizer.zero_grad()  # remove the gradients (for parameter update)

        pred = model(X)  # prediction

        loss = loss_fn(pred, y)  # prediction error

        loss.backward()  # compute all gradients with backpropagation

        optimizer.step()  # parameter update (optimize the model)

        if not quiet and batch % 20 == 0:  # just to show partial information
            done = (batch + 1) * len(X)  # number of samples processed
            total = len(dataloader.dataset)  # total number of samples
            print('  [ %4d / %d ]  loss: %.2f' % (done, total, loss))


# test
def test(dataloader, model, loss_fn):
    """Return model's correct, total and loss when applied to the test data."""
    model.eval()  # set model in evaluation mode

    loss = 0.0  # total loss from all the batches

    # Number of correct guesses and totals per label.
    n_labels = len(dataloader.dataset.labels)
    ncorrect = [0] * n_labels
    ntotal = [0] * n_labels
    
    # For precision calculation later
    true_positives = [0] * n_labels  # Correctly identified as class i
    predicted_totals = [0] * n_labels  # Total predicted as class i
    
    with no_grad():  # we are not training, so no need to compute the gradient
        for X, y in dataloader:  # this dataloader should give only test data
            X, y = X.to(device), y.to(device)  # move to computing device
            pred = model(X)  # model prediction ("guess")
            loss += loss_fn(pred, y)
            
            # Process each prediction
            for pred_i, y_i in zip(pred.cpu().numpy(), y.cpu().numpy()):
                pred_class = pred_i.argmax()
                true_class = y_i
                
                # Count prediction for the predicted class
                predicted_totals[pred_class] += 1
                
                # Count correct predictions
                if pred_class == true_class:
                    ncorrect[true_class] += 1
                    true_positives[pred_class] += 1
                
                # Total count for each true class
                ntotal[true_class] += 1
    
    loss_avg = loss / len(dataloader)  # loss divided by number of batches
    
    return ncorrect, ntotal, true_positives, predicted_totals, loss_avg



# help
help_using_saved_model = """
To use the model (saved in 'model.pt' for example) from python:

  import torch
  from ko_nn import ko_ProstT5_NN

  model = torch.load('model.pt')  # load the file with the saved model

  data = ...  # data like the one from the dataloader
  prediction = model(data)
"""


# main
def main():
    global device

    args = get_args()

    device = args.device if args.device != 'auto' else (
        'cuda' if cuda.is_available() else 'cpu')
    log = print if not args.quiet else (lambda *args: None)
    log('Using computing device:', device)

    log('Reading training and test data...')
    train_dloader, test_dloader, ko_weights = get_dataloaders(
        umap_file=args.samples,
        labels_file=args.labels,
        test_size=args.test_size,
        batch_size=args.batch_size,
        shuffle=args.shuffle)

    ko_weights = ko_weights.to(device)   # You changed this
    log('Creating neural network (model)...')
    nvalues = len(train_dloader.dataset[0][0])  # number of values per sample
    nkos = len(train_dloader.dataset.labels)  # number of possible kos
    print(nkos)
    model = ko_ProstT5_NN(nvalues, args.hidden_sizes, nkos)
    model = model.to(device)  # move model to computing device

    log('Defining how to measure deviations from the correct answer...')
    loss_fn = nn.CrossEntropyLoss(weight=ko_weights)  # loss function

    log('Setting how to use the resulting gradients to optimize the model...')
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    log(f'Training neural network for {args.epochs} epochs...\n')
    for i in range(args.epochs):  # repeat training for a number of "epochs"
        log(f'Epoch {i+1}')

        train(train_dloader, model, loss_fn, optimizer, args.quiet)

        if not args.quiet:
            ncorrect, ntotal, _, _, loss_avg = test(test_dloader, model, loss_fn)
            accuracy = sum(ncorrect) / sum(ntotal)
            log('Accuracy: %.1f %%, avg loss: %g\n' % (100*accuracy, loss_avg))

    # Compute metrics
    ncorrect, ntotal, true_positives, predicted_totals, _ = test(test_dloader, model, loss_fn)
    metric_name = f"{args.output}_" if args.output is not None else ""

    # Write metrics to file
    with open(f"{metric_name}acc_per_class.txt", "w") as f:
        f.write('Per class metrics:\n')
        f.write('  Correct / Total    Recall    Precision    Label\n')
        per_class_recall = []
        per_class_precision = []
    
        for nc, nt, tp, pt, label in zip(ncorrect, ntotal, true_positives, predicted_totals, train_dloader.dataset.labels):
            # Recall = correct predictions for class / total samples of class
            recall = nc / nt if nt > 0 else 0  
        
            # Precision = true positives for class / total predictions for class
            precision = tp / pt if pt > 0 else 0.0
        
            per_class_recall.append(recall)
            per_class_precision.append(precision)
        
            f.write('  %5d / %-5d   %6.3f    %6.3f      %s\n' % (nc, nt, recall, precision, label))
    
        # Calculate weighted metrics
        total_samples = sum(ntotal)
        total_predictions = sum(predicted_totals)
    
        # Global Recall:
        recall = sum(r * nt for r, nt in zip(per_class_recall, ntotal)) / total_samples if total_samples > 0 else 0
    
        # Precision: 
        precision = sum(per_class_precision) / len(per_class_precision) if per_class_precision else 0

	# Weighted precision:
        weighted_precision = sum(p * pt for p, pt in zip(per_class_precision, predicted_totals)) / total_predictions if total_predictions > 0 else 0
    
        f.write('\nGlobal Recall:    %.4f\n' % recall)
        f.write('\nGlobal Precission:    %.4f\n' % precision)
        f.write('Weighted Global Precision: %.4f\n' % weighted_precision)

    print(f"Saving metrics per class in {metric_name}acc_per_class.txt...\n")
    print('Final global recall: %.2f %%' % (100 * recall))
    print('Final global precision: %.2f %%' % (100 * precision))
    print('Final weighted precision: %.2f %%' % (100 * weighted_precision))

    if args.output:
        print(f'\nSaving model to {args.output} ...')
        save(model, args.output)
        log(help_using_saved_model)


# Execute whole script
if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        sys.exit(e)
