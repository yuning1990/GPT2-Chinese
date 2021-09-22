import transformers
import torch
import os
import json
import random
import numpy as np
import argparse

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn import DataParallel
from generate import Generate
from tokenizations.bpe_tokenizer import get_encoder
'''
一边训练，每5个输出1个sample
'''


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', '[SEP]') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')

def get_total_steps_and_all_tokens(tokenized_data_path, num_pieces):
    all_tokens = {}
    full_len = 0
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            tokens = f.read().strip().split()
            full_len += len(tokens)
            all_tokens[i] = [int(token) for token in tokens]
    return full_len, all_tokens

def get_samples(tokens, n_ctx, stride):
    start_point = 0
    samples = []
    while start_point < len(tokens) - n_ctx:
        samples.append(tokens[start_point: start_point + n_ctx])
        start_point += stride
    if start_point < len(tokens):
        samples.append(tokens[len(tokens)-n_ctx:])
    random.shuffle(samples)
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--per_num_epochs_save_models', default=5, type=int, required=False, help='每过几个循环保存一下模型')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)

    model = model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    print('calculating total steps')

    full_len, all_tokens = get_total_steps_and_all_tokens(tokenized_data_path, num_pieces)
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    loss_l = []
    v_loss_l = []
    
    for epoch in range(epochs):
        model_path = output_dir + '/model_epoch{}'.format(epoch + 1)
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        train = x[: int(len(x)/5*4)]
        test = x[int(len(x)/5*4):]
        piece_num = 0
        
        ## begin each epoch train
        model.train()
        for i in train: # 4/5用来train，1/5用来validate
            print("begin epoch {}, {}th txt train".format(epoch + 1, i))
            tokens = all_tokens[i]
            samples = get_samples(tokens, n_ctx, stride)
            for step in range(len(samples) // batch_size):  # drop last
                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, _ = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    now_time = '{}:{}'.format(datetime.now().hour, datetime.now().minute)
                    now_step = step + 1
                    now_piece = piece_num
                    now_epoch = epoch + 1
                    now_loss = running_loss * gradient_accumulation / (log_step / gradient_accumulation)
                    print('Now time: {}, Step {} of piece {} of epoch {}, Loss {}'.format(
                        now_time,
                        now_step,
                        now_piece,
                        now_epoch,
                        now_loss
                        ))
                    loss_l.append({'now_time': now_time,\
                        'step': now_step, 'piece': now_piece,\
                        'epoch': now_epoch, 'loss': now_loss})
                    running_loss = 0
                overall_step += 1
            piece_num += 1
        
        ## validate
        model.eval()
        with torch.no_grad():
            v_loss = 0
            v_steps = 0
            for i in test:
                print("begin epoch {}, {}th txt test".format(epoch + 1, i))
                tokens = all_tokens[i]
                samples = get_samples(tokens, n_ctx, stride)
                for step in range(len(samples) // batch_size):  # drop last
                    #  prepare data
                    batch = samples[step * batch_size: (step + 1) * batch_size]
                    batch_inputs = []
                    for ids in batch:
                        int_ids = [int(x) for x in ids]
                        batch_inputs.append(int_ids)
                    batch_inputs = torch.tensor(batch_inputs).long().to(device)
                    # for j in len(batch_inputs-1):
                    outputs = model(input_ids=batch_inputs, labels=batch_inputs)
                    v_loss += outputs[0].item()
                    #  get loss
                    if multi_gpu:
                        loss = loss.mean()
                    v_steps += 1

            v_loss = v_loss / v_steps
            now_time = '{}:{}'.format(datetime.now().hour, datetime.now().minute)
            print('Now time: {}, Epoch {}, Loss {}, Validation Loss {}'.format(
                now_time,
                now_epoch,
                now_loss,
                v_loss
                ))
            v_loss_l.append({'now_time': now_time,\
                'epoch': now_epoch, 'loss': now_loss,
                'v_loss': v_loss})

        ## sample
        if now_epoch % args.per_num_epochs_save_models == 0: # 每5个epoch出一次sample
            print('saving model for epoch {}'.format(now_epoch))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            
            args.model_path = model_path
            args.key = output_dir.split('_')[-1]
            args.length = 512
            args.nsamples = 1
            args.temperature = 1.0
            args.topk = 40
            args.topp = 0
            args.repetition_penalty = 1.05
            args.save_samples = True
            args.fast_pattern = False
            
            t = str(datetime.now())
            d = ''.join('_'.join(''.join(t.split(":")[:-1]).split(' ')).split('-'))
            args.save_samples_path = 'result_{}/{}_v{}/'.format(args.key, d, now_epoch)
            # print('args.key===', args.key)
            if 'intro' in args.key:
                args.prefix = '[MASK][姓名]：小闹闹\n[年龄]：27\n[省份]：山东\n[大学]：西贝大学\n[学位]：本科\n[专业]：生物\n[成绩]：全班第一\n[荣誉]：三好学生\n[入职行业]：生物制药\n[正文]：'
            
            Generate().run(args)
            # 及时存一下loss
            filename = args.save_samples_path + 'loss.json'
            with open(filename, "w", encoding="utf8") as file:
                json.dump(loss_l, file, indent=2, ensure_ascii=False)
            with open(args.save_samples_path + 'v_loss.json', "w", encoding="utf8") as file:
                json.dump(v_loss_l, file, indent=2, ensure_ascii=False)
 
        print('epoch {} finished'.format(now_epoch))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))
        
        
    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
