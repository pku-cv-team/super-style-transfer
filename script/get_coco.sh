#!/bin/bash

wget http://images.cocodataset.org/zips/train2017.zip -P data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco
wget http://images.cocodataset.org/zips/test2017.zip -P data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco
unzip data/coco/*.zip
rm data/coco/*.zip
