#!/usr/bin/env bash


python3 src/utils/mixing.py \
                --inputdir '<path_to_picutures>' \
    			--outputdir data/ica/ \
    			--samples 2 \
    			--nr-samples 5 \
    			--seed 42 \
    			--time 3 \
    			--mix-type nonlinear \
    			--activation-fun neural_net
