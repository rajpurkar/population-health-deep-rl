if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. First arg should be folder"
    exit
fi
python load/process.py $1
python load/postprocess.py $1/processed.csv
python load/top_k.py $1/post-processed.csv -k 5
python load/top_k.py $1/post-processed.csv -k 10
python load/top_k.py $1/post-processed.csv -k 20