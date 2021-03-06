#!/bin/bash

PACKAGE_DIR=regularized_evolution

mkdir -p $PACKAGE_DIR

pip install --upgrade Pillow torch torchvision numpy graphviz torchsummary seaborn matplotlib --target $PACKAGE_DIR

cp ../../*.py $PACKAGE_DIR
cp -r ../../modules $PACKAGE_DIR
mkdir -p $PACKAGE_DIR/regularized_evolution
cp ../../regularized_evolution/*.py $PACKAGE_DIR/regularized_evolution

cp pnas_job.sh $PACKAGE_DIR

tar cfvz regularized_evolution.tar.gz $PACKAGE_DIR

rm -r $PACKAGE_DIR
