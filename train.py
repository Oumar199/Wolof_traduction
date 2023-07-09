from wolof_translate.utils.training import train
import pytorch_lightning as lt
import argparse
import logging
import json
import sys
import os

# set the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# -----------------------------------------------
# main function
def main(args):
    
    # seed everything
    lt.seed_everything(args.random_state)
    
    # recuperate the configurations
    config = {
        'epochs': args.epochs,
        'log_step': args.log_step,
        'metric_for_best_model': args.metric,
        'metric_objective': args.metric_objective,
        'corpus_1': args.corpus_1,
        'corpus_2': args.corpus_2,
        'train_file': args.training['train'],
        'test_file': args.training['validation'],
        'drop_out_rate': args.dropout,
        'd_model': args.d_model,
        'n_head': args.n_heads,
        'dim_ff': args.d_ff,
        'n_encoders': args.n_encoders,
        'n_decoders': args.n_decoders,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'char_p': args.char_p,
        'word_p': args.word_p,
        'end_mark': args.end_mark,
        'label_smoothing': args.label_smoothing,
        'max_len': args.max_length,
        'random_state': args.random_state,
        'boundaries': args.boundaries.split(','),
        'batch_sizes': args.batch_sizes.split(','),
        'batch_size': args.batch_size, 
        'warmup_init': args.warmup_init,
        'relative_step': args.relative_step,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        # --------------------> Must be changed when continuing a training
        'model_dir': args.training['model'],
        'new_model_dir': os.path.join(args.model_dir, args.new_model_dir_),
        'continue': args.continue_, # --------------------------> Must be changed when continuing training
        'logging_dir': os.path.join(args.output_data_dir, args.logging_dir),
        'save_best': args.save_best,
        'tokenizer_path': args.training['tokenizer'],
        'version': args.version,
        # in the case of a distributed training
        'backend': args.backend,
        'hosts': args.hosts,
        'current_host': args.current_host,
        'num_gpus': args.num_gpus,
        'logger': logger,
        'return_trainer': False,
        'include_split': False,
    }
    
    # train the model
    train(config)

# -----------------------------------------------
# main part arguments and passing the arguments

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    
    # epochs
    parse.add_argument('--epochs', type=int, default=200, help='Number of epochs', metavar='N')
    
    # log step
    parse.add_argument('--log_step', type=int, default=10, help='Print log every log_step', metavar='ls')
    
    # metric for best model
    parse.add_argument('--metric', type=str, default='bleu', help='Metric for best model', metavar='m')
    
    # metric objective
    parse.add_argument('--metric_objective', type=str, default='maximize', help='Metric objective', metavar='mo')
    
    # corpus_1
    parse.add_argument('--corpus_1', type=str, default='french', help='Path to first language', metavar='c1')
    
    # corpus_2
    parse.add_argument('--corpus_2', type=str, default='wolof', help='Path to second language', metavar='c2')
    
    # dropout
    parse.add_argument('--dropout', type=float, default=0.2, help='Dropout rate', metavar='dr')
    
    # d_model
    parse.add_argument('--d_model', type=int, default=512, help='Dimension of model', metavar='dm')
    
    # n_heads
    parse.add_argument('--n_heads', type=int, default=8, help='Number of heads', metavar='nh')
    
    # d_ff
    parse.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed forward', metavar='dff')
    
    # n_encoders
    parse.add_argument('--n_encoders', type=int, default=6, help='Number of encoders', metavar='ne')
    
    # n_decoders
    parse.add_argument('--n_decoders', type=int, default=6, help='Number of decoders', metavar='nd')
    
    # learning rate
    parse.add_argument('--lr', type=float, default=None, help='Learning rate', metavar='lr')
    
    # weight_decay
    parse.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay', metavar='wd')
    
    # char p
    parse.add_argument('--char_p', type=float, default=0.9, help='Char p', metavar='cp')
    
    # word p
    parse.add_argument('--word_p', type=float, default=0.2, help='Word p', metavar='wp')
    
    # end mark
    parse.add_argument('--end_mark', type=int, default=3, help='End mark', metavar='em')
    
    # label smoothing
    parse.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing', metavar='ls')
    
    # max length
    parse.add_argument('--max_length', type=int, default=20, help='Max length', metavar='ml')
    
    # random state
    parse.add_argument('--random_state', type=int, default=0, help='Random state', metavar='rs')
    
    # boundaries
    parse.add_argument('--boundaries', type=str, default='2,31,59,87,115,143,171', help='Buckets boundaries', metavar='b')
    
    # batch sizes
    parse.add_argument('--batch_sizes', type=str, default='256,128,64,32,16,8,4,2', help='Batch sizes', metavar='bs')
    
    # batch size
    parse.add_argument('--batch_size', type=int, default=256, help='Batch size', metavar='bs')
    
    # warmup init
    parse.add_argument('--warmup_init', type=bool, default=True, help='Warmup init', metavar='wi')
    
    # relative step
    parse.add_argument('--relative_step', type=bool, default=True, help='Relative step', metavar='rs')
    
    # num workers
    parse.add_argument('--num_workers', type=int, default=1, help='Number of workers', metavar='nw')
    
    # pin memory
    parse.add_argument('--pin_memory', type=bool, default=True, help='Pin memory', metavar='pm')
   
    # new model dir
    parse.add_argument('--new_model_dir_', type=str, default=f'custom_transformer_v6_fw', help='New model directory', metavar='nmd')
    
    # continue *****
    parse.add_argument('--continue_', type=bool, default=False, help='Continue', metavar='c')
    
    # logging dir
    parse.add_argument('--logging_dir', type=str, default='custom_transform_fw', help='Logging directory', metavar='ld')
    
    # save best
    parse.add_argument('--save_best', type=bool, default=True, help='Save best', metavar='sb')
    
    # version
    parse.add_argument('--version', type=int, default=6, help='Version', metavar='v')
    
    # backend
    parse.add_argument('--backend', type=str, default='nccl', help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)', metavar='bk')
    
    # hosts
    parse.add_argument('--hosts', type=list, default=json.loads(os.environ["SM_HOSTS"]), help='Hosts', metavar='h')
    
    # current host
    parse.add_argument('--current_host', type=str, default=os.environ["SM_CURRENT_HOST"], help='Current host', metavar='ch')
    
    # model dir
    parse.add_argument('--model_dir', type=str, default=os.environ["SM_MODEL_DIR"], help='Model directory', metavar='md')
    
    # output data dir
    parse.add_argument('--output_data_dir', type=str, default=os.environ["SM_OUTPUT_DATA_DIR"], help='Output data directory', metavar='od')
    
    # num gpus
    parse.add_argument('--num_gpus', type=int, default=os.environ["SM_NUM_GPUS"], help='Number of gpus', metavar='ng')
    
    # training dir
    parse.add_argument('--training', type=json.loads, default=json.loads(os.environ['SM_CHANNEL_TRAINING']), help='Training data paths', metavar='td')
    
    # pass the arguments to the main function
    args = parse.parse_args()
    main(args)