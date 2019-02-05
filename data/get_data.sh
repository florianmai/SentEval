#
# Call SentEval downstream task script and glue script
#


#SentEval
cd downstream
bash get_transfer_data.bash


# GLUE
cd ../GLUE

# download MRPC dataset
wget https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi

# download everything else
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
