## Joint_biased_embeddings

This is the codebase to run the experiments described in paper 'Improving Knowledge Graph Embeddings with Inferred Entity Types'
train.py provides a command-line interface to run the experiments.
test.py runs 2 training epochs on various combinations of the parameters to
check that the code runs.

Main files are in src/:
experiment.py - Loads the dataset and constructs the data structures for running
the experiments.
modules.py - Contains several classes for models used in experiment. Joint_Module
is the module for joint training, and needs to be passed submodules.
trainer.py - Contains the Trainer object that is responsible for training the modules.
batching.py - Responsible for generating batches and negative examples to feed
to the Trainer

datasets can be obtained from:
FB15k-237 and YAGO3-10: https://github.com/TimDettmers/ConvE
FB15K: https://everest.hds.utc.fr/doku.php?id=en:transe


## License

This library is licensed under the Apache 2.0 License. 
