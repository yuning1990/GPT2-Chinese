from generate import *
from datetime import datetime

def main():
    ''' e.g. 
    python ./generate.py --length=512 
    --nsamples=1 
    --prefix=[MASK]哈利站在窗边 
    --tokenizer_path cache/vocab_small.txt
    --topk 40 --model_path model/model_epoch29 
    --save_samples --save_samples_path result/20210915_29_1135 
    --model_config model/model_epoch29/config.json --repetition_penalty 1.05 --temperature 1.1
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', default='intro', type=str, required=False, help='哪个模型')
    parser.add_argument('--model_v', default='-1', type=str, required=False, help='第几个模型')
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=1024, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=1, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1.1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=20, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='哈利站在窗边', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', default=True, help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.05, type=float, required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    
    if args.model_v != '-1':
        args.model_path = '{}/model_epoch{}'.format(args.model_path.split('/')[0], args.model_v)
    else:
        args.model_path = args.model_path

    t = str(datetime.now())
    d = ''.join('_'.join(''.join(t.split(":")[:-1]).split(' ')).split('-'))
    args.save_samples_path = 'result_{}/{}_v{}'.format(args.key, d, args.model_v)
    Generate().run(args)

if __name__ == '__main__':
    main()
