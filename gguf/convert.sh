echo "Running this script assumes you have llama.cpp installed with all its requirements."
echo "Usage: convert.sh model_name local_location llamacpp_location llamacpp_outfile llamacpp_outtype"
python download_model.py $1 $2
python $3/convert.py $2 --outfile $4 --outtype $5
