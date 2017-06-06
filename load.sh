if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. First arg should be folder"
    exit
fi
python load/process.py $1
python load/postprocess.py $1/processed.csv