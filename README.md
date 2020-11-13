# ElectronDensity2

## The workflow for preparing the data
1. run input.qm.parse_dataset to extract data and generate electron densities
2. run input.tokenizer.Tokenizer with initialize_from_dataset=True to initialize tokenizer
3. run input.tfrecords.train_validation_test_split to create and split data into traing validation and test   set tfrecords

## Getting the data
run input.tfrecords.input_fn to create tfrecords reader

##Training models and doing the translation to SMILES
In the examples folder there jupyter notebooks with this


