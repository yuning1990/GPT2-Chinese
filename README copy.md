# 每个result参数
python ./generate.py --length=512 --nsamples=3 --prefix=在霍格沃茨学校新生宴会上， --fast_pattern --tokenizer_path cache/vocab_user.txt --topk 40 --model_path model/model_epoch72 --save_samples --segment --save_samples_path result/20201207_v2

python ./generate.py --length=512 --nsamples=3 --prefix=在霍格沃茨学校新生宴会上， --fast_pattern --tokenizer_path cache/vocab_user.txt --topk 40 --model_path model/model_epoch96 --save_samples --segment --save_samples_path result/20201209_v1


python ./generate.py --length=512 --nsamples=3 --prefix=哈利说， --fast_pattern --tokenizer_path cache/vocab_user.txt --topk 40 --model_path model/model_epoch2 --save_samples --save_samples_path result/20210913_v1

python ./generate.py --length=512 --nsamples=1 --prefix=[CLS]我叫李焕英， --tokenizer_path cache/vocab.txt --topk 40 --model_path sample/general_ch --save_samples --save_samples_path result/general_ch --temperature 1 --model_config sample/general_ch/config.json --tokenizer_path sample/general_ch/vocab.txt

# train参数
python train_single.py --raw --pretrained_model sample/general_ch --raw_data_path data/train.json --num_pieces 100 --batch_size 2 --epochs 10 --tokenizer_path sample/general_ch/vocab.txt

python train.py --raw --raw_data_path data/train.json --num_pieces 36 --batch_size 1 --epochs 50 --tokenizer_path cache/vocab_user.txt


python ./generate.py --length=512 --nsamples=1 --prefix=[CLS]哈利波特说， --fast_pattern --tokenizer_path cache/vocab_user.txt --topk 40 --model_path model/model_epoch18 --save_samples --save_samples_path result/20210304_v1 --model_config model/model_epoch18/config.json


