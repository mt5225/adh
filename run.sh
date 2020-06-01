export TRAINING_DATA=./input/train_folds.csv
export TEST_DATA=./input/test.csv
export FOLD=0
export MODEL=$1

# training
# python -m src.train

# predict
python -m src.predict