#!/usr/bin/env bash

python classifier.py --option=finetune --batch_size=8 --use_gpu --train=data/cfimdb-train.txt --dev=data/cfimdb-dev.txt --test=data/cfimdb-test.txt --dev_out=cfimdb-dev-output.txt --test_out=cfimdb-test-output.txt || exit 1
mv finetune-10-1e-05.pt cfimdb.pt

python classifier.py --option=finetune --batch_size=64 --use_gpu --train=data/sst-train.txt --dev=data/sst-dev.txt --test=data/sst-test.txt --dev_out=sst-dev-output.txt --test_out=sst-test-output.txt
mv finetune-10-1e-05.pt sst.pt
