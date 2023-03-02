from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from models.finetune_model import NPairContrastiveFinetuneEncoder
from models.pretrained_encoder import PretrainedSentenceEncoder
from losses.multiple_negative_ranking_loss import MultipleNegativesRankingLoss
from data_utils.nli_data import AllNliTripletTextDataModule

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    seed_everything(args.seed)

    dataset_name = args.dataset_name
    # dataset names should be AG_NEWS, IMDB

    data_module = AllNliTripletTextDataModule(
        dataset_name, batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    config = {}
    config["num_classes"] = data_module.label_size
    config["lr"] = args.lr
    encoder_model = PretrainedSentenceEncoder(
        pretrained_model_name=args.pretrained_model_name,
        truncate=args.truncate,
        pooling_method=args.pooling_method,
        normalize_embeddings=args.normalize_embeddings,
        num_layers=args.num_layers,
    )
    loss = MultipleNegativesRankingLoss()
    model = NPairContrastiveFinetuneEncoder(encoder=encoder_model, contrast_loss=loss)

    checkpoint_callback = ModelCheckpoint(
        monitor="Cosine-Similarity-Spearman_epoch",
        save_top_k=2,
        mode="max",
        every_n_train_steps=args.val_step_interval,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    project_name = "sentence_bert_finetune"
    tags = [args.pretrained_model_name, args.dataset_name, "contrastive"]
    if args.num_layers > 0:
        tags.append(f"num_layers:{str(args.num_layers)}")
    wandb_logger = WandbLogger(
        project=project_name,  # group runs in "MNIST" project
        log_model="all" if args.log_model else False,
        save_dir=args.default_root_dir,
        group=args.dataset_name,
        tags=tags,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate),
            checkpoint_callback,
            lr_monitor,
        ],
        deterministic=True,
        val_check_interval=args.val_step_interval,
    )

    print(dataset_name)

    trainer.validate(model, dataloaders=val_dataloader)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--val_step_interval", type=int, default=500)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--log_model", action="store_true")

    # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--truncate", type=int, default=128)
    parser.add_argument("--pooling_method", type=str, default="mean_pooling")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
