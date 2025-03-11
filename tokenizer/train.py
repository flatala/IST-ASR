#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with librispeech.
The tokenizer converts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.

To run this recipe, do the following:
> python train.py hyperparams/bpe_1k.yaml


Authors
 * Abdel Heba 2021
"""

import sys

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Train tokenizer
    hparams["tokenizer"]()