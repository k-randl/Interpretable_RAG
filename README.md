# Usage
After cloning or downloading this repository, first run the Linux shell script [`./setup.sh`](https://github.com/k-randl/self-explaining_llms/blob/main/setup.sh).
It will initialize the workspace by performing the following steps:

1. It will install the required Python modules by running `pip install -r "./requirements.txt"`
2. It will download the necessary Python code to compute the [BARTScore](https://github.com/neulab/BARTScore) by Yuan et al. (2021) to "./resources/bart_score.py".
3. It will download the necessary Python code to compute the [BARTScore](https://github.com/jbshp/LongDocFACTScore) by Bishop et al. (2024) to "./resources/ldfacts.py".

After running the script, copy your credential file to "./data/service-account-external-efra.json"