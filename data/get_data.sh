#
# Call SentEval downstream task script and glue script
#


#SentEval
cd downstream
bash get_transfer_data.bash

# GLUE
cd ../GLUE
python download_glue_data.py

# do some renaming and merging on the data.
mv glue_data/* .

# replace MRPC dataset
rm -rf ../downstream/MRPC/
mv MRPC ../downstream/MRPC

# prepare MNLI dataset
mv MNLI/original/* MNLI/
rm -rf original
cd MNLI
cp multinli_1.0_dev_matched.txt multinli_1.0_dev_both.txt
tail -n 10000 multinli_1.0_dev_mismatched.txt >> multinli_1.0_dev_both.txt
