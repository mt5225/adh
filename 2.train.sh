export TRAINING_DATA=./input/cat_II/train_folds.csv
export TEST_DATA=./input/cat_II/test.csv
# FOLD can be 0,1,2,3,4
export FOLD=3
export MODEL=extratrees

# start training
python -m src.train
