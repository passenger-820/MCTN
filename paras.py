import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # path to dataset (need to be pytorch files, like .pth)
    parser.add_argument('--mobiact-dir', default=r"path\to\dataset")
    parser.add_argument('--sisfall-dir', default=r"path\to\dataset")
    parser.add_argument('--softfall-dir', default=r"path\to\dataset")

    # downsample from 200Hz.    equals to 200/default
    parser.add_argument('--sisfall-down', default=8, type=int)
    # downsample from 100Hz.    equals to 100/default
    parser.add_argument('--mobiact-down', default=4, type=int)

    # data window
    parser.add_argument('--pre-len', default=15, type=int, help='lead time')
    parser.add_argument('--front-len', default=75, type=int, help='before SMV')
    parser.add_argument('--rear-len', default=0, type=int, help='after SMV')

    # sensor number
    parser.add_argument('--modal-two', default=2, type=int)
    parser.add_argument('--modal-three', default=3, type=int)

    # for PE
    parser.add_argument('--d-pe', default=76, type=int)

    # for Transformer Encoder
    parser.add_argument('--d-model', default=77, type=int)
    parser.add_argument('--n-head', default=7, type=int)
    parser.add_argument('--dim-feedforward', default=16, type=int)
    """
    Hz       downsample  total     n_head        pre    front
    20Hz        10/5      63          7           12      61      
    25Hz        8/4       77          7           15      75
    50Hz        4/2       145         5           30      143
    100Hz       2/1       297         9           60      295         
    200Hz       1         583         11          120     581         
    """

    # for projection and prediction layer
    parser.add_argument('--projected-dim', default=8, type=int)
    parser.add_argument('--predicted-dim', default=2, type=int)

    # experiment
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    return parser.parse_args()

