#/bin/bash

kaggle competitions download -c rsna-knee-abnormality-detection -p data/raw
unzip data/raw/rsna-knee-abnormality-detection.zip -d data/raw