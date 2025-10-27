#!/bin/sh

# install requirements:
pip install -r ./requirements.txt

# download bart score:
mkdir ./src/longdocfactscore
wget -P ./src/longdocfactscore/ https://raw.githubusercontent.com/neulab/BARTScore/main/bart_score.py

# download ldfacts score:
wget -P ./src/longdocfactscore/ https://raw.githubusercontent.com/jbshp/LongDocFACTScore/refs/heads/main/src/longdocfactscore/ldfacts.py

mkdir ./data
python -m spacy download en_core_web_sm