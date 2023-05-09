import argparse

import config
from train import Trainer
from evaluate import Tester
from predict import predict
from dataset import build_dataloaders

# TODO: change everything
    
def main(args):

    if args.subparser == "train":
        trainer = Trainer(args)
        train_dl, val_dl, _ = build_dataloaders(trainer.hprms, config.DATA_DIR) 
        trainer.train(train_dl, val_dl)
        
    elif args.subparser == "evaluate":
        tester = Tester(args)
        _, _,  test_dl = build_dataloaders(tester.hprms, config.DATA_DIR)
        tester.evaluate(test_dl)   
             
    elif args.subparser == "predict":
        predict(args)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='U-Net Speech Enhancer')
    subparsers = parser.add_subparsers(help='Help for subcommand', dest="subparser")
    
    # Train commands
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--experiment_name', 
                              type=str, 
                              help='Choose a name for the experiment',
                              default='enhancerHZ_00') 
    
    parser_train.add_argument('--model', 
                              type=str, 
                              help="The model you want to train",
                              choices=['enhancer', 'enhancer_hz', 'melspec2spec'],
                              default='enhancer_hz') 

    parser_train.add_argument('--enhancer', 
                              type=str, 
                              help="Name of the directory containing the enhancer's weights and results. \
                                  To be used only if training melspec2spec",
                              default=None)
    
    parser_train.add_argument('--resume_training',
                              action='store_true',
                              help="Use this flag if you want to restart training from a checkpoint")
    
    parser_train.add_argument('--overwrite',
                              action='store_true', 
                              help="Use this flag if you want to overwrite an experiment")

    # Evaluate commands
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_eval.add_argument('--experiment_name', 
                             type=str, 
                             help="The name of the experiment, can be the enhancer or the melspec2spec directory",
                             default='enhancer80_00') 
    parser_eval.add_argument('--pinv', 
                             type=str, 
                             help="Name of the directory containing the melspec2spec's weights and results.\n \
                                  Not to be used with a standard STFT enhancer",
                             default=None) 
    parser_eval.add_argument('--freq', 
                             type=str,
                             choices=['mel', 'hz'],
                             help="mel or hz model", 
                             default='hz') 
    
    # Predict commands
    parser_predict = subparsers.add_parser('predict', 
                                     help='Use the model for prediction')
    parser_predict.add_argument('--enhancer', 
                                type=str, 
                                help="Name of the directory containing the enhancer's weights and results",
                                default='enhancer80_00') 
    
    parser_predict.add_argument('--pinv', 
                                type=str, 
                                help="Name of the directory containing the melspec2spec's weights and results",
                                default='pinvdae80_00') 
    
    parser_predict.add_argument('--audio', 
                                type=str,
                                help="audio.wav in mixture_example/", 
                                default='noisy0.wav')


    args = parser.parse_args()
    
    main(args)