import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(ADAM_loss_hist, SGD_loss_hist, SGD_LR_SCH_loss_hist, plot_path):
      # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(ADAM_loss_hist,'r',linewidth=2.0)
    plt.plot(SGD_loss_hist,'b',linewidth=2.0)
    plt.plot(SGD_LR_SCH_loss_hist,'v',linewidth=2.0)
    plt.legend(['RNN ppl', 'GRU ppl', 'TRANSFORMER ppl'],fontsize=18)
    plt.xlabel('Time-step (t) ',fontsize=16)
    plt.ylabel('Perplexity',fontsize=16)
    plt.title('Perplexity Curves of Multiple Architecture',fontsize=20, fontweight='bold')
    plt.savefig(plot_path)
  
def plot_training_history_wc(dir, ADAM_loss_hist, SGD_loss_hist, SGD_LR_SCH_loss_hist, plot_path):
      # Loss Curves
    plt.figure(figsize=[8,6])
    rnn = []
    gru = []
    tra = []
    if dir == 'adam_plot':
      for x in range(40):
            rnn.append(x * 176.53509736061096)
            gru.append(x * 244.4788110256195)
            tra.append(x * 58.293203830718994)
    elif dir == 'sgd_plot':
      for x in range(40):
            rnn.append(x * 170.41701912879944)
            gru.append(x * 236.4971776008606)
            tra.append(x * 122.8176200389862)
    elif dir == 'sch_plot':
      for x in range(40):
            rnn.append(x * 97.27239632606506)
            gru.append(x * 236.47585034370422)
            tra.append(x * 161.07048439979553)
    plt.plot(rnn, ADAM_loss_hist,'r',linewidth=2.0)
    plt.plot(gru, SGD_loss_hist,'b',linewidth=2.0)
    plt.plot(tra, SGD_LR_SCH_loss_hist,'v',linewidth=2.0)
    plt.legend(['RNN ppl', 'GRU ppl', 'TRANSFORMER ppl'],fontsize=18)
    plt.xlabel('Wall-Clock time (t) ',fontsize=16)
    plt.ylabel('Perplexity',fontsize=16)
    plt.title('Perplexity Curves of Multiple Architecture',fontsize=20, fontweight='bold')
    plt.savefig(plot_path)

## READ_ME: s_exec python plot.py --save_dir=RNN_4-1

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='RNN',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic, 
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]


dirs = ['adam_plot', 'sgd_plot', 'sch_plot']
y = ['RNN', 'GRU', 'TRANSFORMER']
for dir in dirs:
  x = dict()
  for name in y:
    lc_path = os.path.join(dir, 'learning_curves_' + name + '.npy')
    plot_path = os.path.join(dir, dir + 'learning_curves_plot_epoch.png')
    x[name] = np.load(lc_path)[()]
  print('\nDONE\n\Load learning curves of ' + lc_path)

  # x = np.load(lc_path)[()]

  epoch = 0
  plot_training_history(x['RNN']['val_ppls'], x['GRU']['val_ppls'], x['TRANSFORMER']['val_ppls'], plot_path)

for dir in dirs:
  x = dict()
  for name in y:
    lc_path = os.path.join(dir, 'learning_curves_' + name + '.npy')
    plot_path = os.path.join(dir, dir + 'learning_curves_plot_WL.png')
    x[name] = np.load(lc_path)[()]
  print('\nDONE\n\Load learning curves of ' + lc_path)

  # x = np.load(lc_path)[()]

  epoch = 0
  plot_training_history_wc(dir, x['RNN']['val_ppls'], x['GRU']['val_ppls'], x['TRANSFORMER']['val_ppls'], plot_path)