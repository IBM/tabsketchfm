import random
from argparse import ArgumentParser
import pytorch_lightning as pl
from tabsketchfm import Tokenizer, PretrainDataModule, TabSketchFM
from transformers import AutoConfig, AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch, numpy as np, random




def auto_lr_find(args, lmmodel, tabular_tokenizer):
    trainer = pl.Trainer.from_argparse_args(args,
                                            accelerator="gpu" if args.gpus else "cpu", devices=1)
    sample_data_module = PretrainDataModule(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tabular_tokenizer,
        data_dir=args.data_dir,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        run_on_sample=args.run_on_sample,
        sample_size = args.sample_size
    )
    lr_finder = trainer.tuner.lr_find(lmmodel, datamodule=sample_data_module, num_training=100)
    print('lr_finder.results:', lr_finder.results)
   
    new_lr = lr_finder.suggestion()
    print('LR suggestion: ', new_lr)
    if new_lr:
        print('Updating LR based on auto finder: ', new_lr)
        lmmodel.hparams.lr = new_lr
        print(lmmodel.hparams.lr)


def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="bert-base-uncased")
    parser.add_argument('--data_dir', type=str,
                        default='../sample_extracted_test_data')
    parser.add_argument('--dataset', type=str, help='name of the file that contains the train test splits for the data')
    parser.add_argument('--run_on_sample', action='store_true', default=False)
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_token_types', type=int, default=5)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=16)
    parser.add_argument('--auto_find_lr', action='store_true', default=False)
    parser.add_argument('--run_local', action='store_true', default=False)
    parser.add_argument('--save_bert_model', action='store_true', default=False)
    parser.add_argument('--bert_model_path', type=str, default='tmp/hf_bert_checkpoint', help="Path to save the trained TabSketchFM model")
    parser.add_argument('--cols_equal', action='store_true', default=False)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--preprocessed_data', type=int, default=1)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TabSketchFM.add_model_specific_args(parser)
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    pl.seed_everything(args.random_seed)
    print("SEEDED ALL")


    # ------------
    # data
    # ------------
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.max_position_embeddings = args.max_seq_length
    toks = AutoTokenizer.from_pretrained(args.model_name_or_path)


    # ------------
    # Data Module
    # ------------


    tokenizer = Tokenizer(tokenizer=toks, config=config)

    data_module = PretrainDataModule(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        dataset=args.dataset,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        run_on_sample=args.run_on_sample,
        sample_size=args.sample_size,
        cols_equal = args.cols_equal,
        concat=True,
        preprocessed_data = bool(args.preprocessed_data==1)
    )

    # ------------
    # model
    # ------------

    model = TabSketchFM(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon
    )

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.0,
        patience=5,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    print('Parameters:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    trainer_ddp = pl.Trainer.from_argparse_args(args,
                    gradient_clip_val=0.5, callbacks=[early_stop_callback, lr_monitor])

    if args.auto_find_lr:
        auto_lr_find(args, model, tokenizer)
        exit(0)

    trainer_ddp.fit(model, data_module)
    if args.save_bert_model:
        model.model.bert.save_pretrained(args.bert_model_path)
        tokenizer.save_tokenizer_checkpoint(args.bert_model_path)
        
    #shows metrics for last epoch only
    print(trainer_ddp.logged_metrics)
    
    
    # ------------
    # testing
    # ------------
    trainer_ddp.test(model, dataloaders=data_module.test_dataloader())


if __name__ == '__main__':
    cli_main()
