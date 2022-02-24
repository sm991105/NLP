#!/bin/bash

set -e
set -x

# Download datafile
mkdir -p data/
cd data/
URL=http://mattmahoney.net/dc/text8.zip
wget $URL
unzip $(basename $URL)
rm text8.zip
