conda activate captum

#python generate_model.py --preprocess --experiment 23March/TCN5
#python generate_model.py --train  --architecture TCN  --epochs 40 --batch_size 300 --experiment 23March/TCN5 --log

#python generate_model.py --preprocess --experiment 23March/SmallTCN
#python generate_model.py --train  --architecture SmallTCN  --epochs 40 --batch_size 300 --experiment 23March/SmallTCN --log

python generate_model.py --preprocess --experiment 23March/TCN4
python generate_model.py --train  --architecture TCN4  --epochs 40 --batch_size 300 --experiment 23March/TCN4 --log

#python generate_model.py --preprocess --experiment 24March/xResNet
#python generate_model.py --train  --architecture xResNet  --epochs 40 --batch_size 300 --experiment 24March/xResNet --log

#python generate_model.py --preprocess --experiment 24March/wangFCN
#python generate_model.py --train  --architecture wangFCN  --epochs 40 --batch_size 300 --experiment 24March/wangFCN --log

#python generate_model.py --preprocess --experiment 24March/LSTM
#python generate_model.py --train  --architecture singh_LSTM  --epochs 40 --batch_size 300 --experiment 24March/LSTM --log

#python generate_model.py --preprocess --experiment 23March/GRU
#python generate_model.py --train  --architecture singh_GRU  --epochs 40 --batch_size 300 --experiment 23March/GRU --log