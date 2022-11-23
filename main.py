import argparse

import config
from training import train_model
from predict import predict
from evaluate import evaluate


def parse_args():
    
    parser = argparse.ArgumentParser(description='U-Net Speech Enhancer',
                                     epilog='I hope it sounds good')
    subparsers = parser.add_subparsers(help='Help for subcommand', dest="subparser")
    
    # Train commands
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('experiment_name', 
                              type=str, 
                              help='Choose a name for your experiment', 
                              default='unet0')
    parser_train.add_argument('--weights_dir', 
                              type=str, 
                              help='If you want to restart the training, specify the weigths location',
                              default=None)
    parser_train.add_argument('--batch_size', 
                              type=int,
                              help='Batch size for training. Def: 16',
                              default=16)
    parser_train.add_argument('--epochs', 
                              type=int,
                              help='Number of epochs to train the model. Def: 30',
                              default=30)
    parser_train.add_argument('--patience', 
                              type=int,
                              help='Patience parameter for early-stopping. Def: 10',
                              default=10)
    parser_train.add_argument('--lr', 
                              type=float,
                              help='Learning Rate for training. Def: 1e-3',
                              default=1e-3)
    # Predict commands
    parser_predict = subparsers.add_parser('predict', 
                                     help='Use the model for prediction')
    parser_predict.add_argument('audio_path',
                                type=str,
                                help='Relative path to .wav audio in mixture_example folder',
                                default='noisy0.wav')
    parser_predict.add_argument('--weights_dir', 
                                type=str, 
                                help='Directory of model weights',
                                default='unet0')

    # Evaluate commands
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_eval.add_argument('--weights_dir',
                            type=str, 
                            help='Directory of model weights',
                            default='unet0')

    args = parser.parse_args()
    return args

    
def main(args):
    if args.subparser == "train":
        train_model(args)
    elif args.subparser == "predict":
        predict(args)
    elif args.subparser == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    args = parse_args() 
    main(args)