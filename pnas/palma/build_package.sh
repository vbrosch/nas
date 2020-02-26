#!/bin/bash

PACKAGE_DIR=pnas

mkdir -p $PACKAGE_DIR

pip install --upgrade Pillow torch torchvision numpy graphviz torchsummary seaborn matplotlib --target $PACKAGE_DIR

cp ../../*.py $PACKAGE_DIR
cp -r ../../modules $PACKAGE_DIR
mkdir -p $PACKAGE_DIR/pnas
cp ../../pnas/*.py $PACKAGE_DIR/pnas

cp pnas_job.sh $PACKAGE_DIR

tar cfvz pnas.tar.gz $PACKAGE_DIR

rm -r $PACKAGE_DIR
