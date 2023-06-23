from transformers import PreTrainedTokenizerFast
from gt4sd_trainer.hf_pl.core import LanguageModelingTrainingPipeline
import json
from pathlib import Path
import os
import click


@click.command()
@click.option("--train_config_path", required=True, help="Data path")

def main(train_config_path: Path):

    with open(train_config_path, 'r') as config_file:
        config = json.load(config_file)

    with open(Path(config['model_args']['model_config_template']).joinpath('config.json'), 'r') as model_config_file:
        model_config = json.load(model_config_file)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config['model_args']['tokenizer'])
    model_config['eos_token_id'] = tokenizer.eos_token_id
    model_config['pad_token_id'] = tokenizer.pad_token_id
    model_config['decoder_start_token_id'] = tokenizer.pad_token_id
    model_config['vocab_size'] = tokenizer.vocab_size

    os.makedirs(config['model_args']['model_config_name'], exist_ok=True)
    with open(Path(config['model_args']['model_config_name']).joinpath('config.json'), 'w') as model_config_file:
        json.dump(model_config, model_config_file, indent=4)

    pipe = LanguageModelingTrainingPipeline()
    pipe.train(**config)




if __name__ == '__main__':
    main()