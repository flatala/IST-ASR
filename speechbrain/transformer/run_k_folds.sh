# echo 'running split 1'
# python train.py hparams/conformer_small_fold_1.yaml

# echo 'running split 2'
# python train.py hparams/conformer_small_fold_2.yaml

echo 'running split 3'
python train.py hparams/conformer_small_fold_3.yaml

echo 'running split 4'
python train.py hparams/conformer_small_fold_4.yaml

echo 'running split 5'
python train.py hparams/conformer_small_fold_5.yaml