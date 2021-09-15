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

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    if args.model_v != '-1':
        model_path = 'model/model_epoch{}'.format(args.model_v)
    else:
        model_path = args.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx

    t = str(datetime.now())
    d = ''.join('_'.join(''.join(t.split(":")[:-1]).split(' ')).split('-'))
    if args.model_v != '-1':
        save_samples_path = 'result/{}_v{}'.format(d, args.model_v)
    else:
        save_samples_path = args.save_samples_path

    if args.save_samples:
        if not os.path.exists(save_samples_path):
            os.makedirs(save_samples_path)
        samples_file = open(save_samples_path + '/samples.txt', 'w', encoding='utf8')
    while True:
        raw_text = args.prefix
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
        generated = 0
        for _ in range(nsamples // batch_size):
            out = generate(
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=length,
                is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
            )
            for i in range(batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                print(info)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                if args.save_samples:
                    samples_file.write(info)
                    samples_file.write(text)
                    samples_file.write('\n')
                    samples_file.write('=' * 90)
                    samples_file.write('\n' * 2)
        print("=" * 80)
        if generated == nsamples:
            # close file when finish writing.
            if args.save_samples:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
