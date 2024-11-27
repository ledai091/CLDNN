import torch
from train import Trainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CLDNN Model Parameters')
    
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--cnn_out_channels', type=int, default=256,
                        help='Number of CNN output channels')
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--lstm_num_blocks', type=int, default=3,
                        help='Number of LSTM blocks')
    parser.add_argument('--lstm_num_cells_per_block', type=int, default=3,
                        help='Number of LSTM cells per block')
    parser.add_argument('--dnn_output_size', type=int, default=2,
                        help='Size of DNN output')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Folder to save model and result')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epoch')
    parser.add_argument('--class_weight', type=bool, default=False,
                    help='Using class weight')
    parser.add_argument('--augmentation', type=bool, default=False,
                help='Using augmentation')
    args = parser.parse_args()
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    args.lstm_input_size = args.cnn_out_channels
    
    return args

def main(args):
    trainer = Trainer(
        args=args
    )
    trainer.train(epochs=args.epochs)
    trainer.evaluate()

if __name__ == '__main__':
    main(parse_args())