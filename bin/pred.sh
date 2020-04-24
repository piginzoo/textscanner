if [ "$1" == "" ] || [ "$1" == "help" ]; then
    echo "命令格式："
    echo "\tpred.sh <image> <model>"
    exit
fi

echo "开始预测"

python -m main.pred $1 $2