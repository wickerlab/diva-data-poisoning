#!bin/bash

# Train SVM and generate Adversarial Label Flip Attack examples
python ./experiments/synth/nn/step2_train_attack.py -f ./data/synth/ -s 0.05 -t 1000 -m 0.41
