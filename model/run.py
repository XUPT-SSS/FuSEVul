import logging
import pandas as pd
import torch
import time
import numpy as np
import random
from model import Code_Note
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig, AutoModel, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup, T5ForConditionalGeneration
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    # 读取训练集和验证集
    def __init__(self, code, text, label):
        self.code = code
        self.text = text
        self.label = label

class InputFeatures(object):
    # 存放训练集和验证集id
    def __init__(self, inputs_code_ids, inputs_code_masks, inputs_text_ids, inputs_text_masks, label):
        self.inputs_code_ids = inputs_code_ids
        self.inputs_code_masks = inputs_code_masks
        self.inputs_text_ids = inputs_text_ids
        self.inputs_text_masks = inputs_text_masks
        self.label = label

def read_file(codefile, textfile):
    # 读取代码和注释文件
    examples = []
    code_data = pd.read_csv(codefile,  na_filter=False, encoding_errors='ignore')
    text_data = pd.read_csv(textfile, na_filter=False, encoding_errors='ignore')
    code = code_data['text'].values.tolist()
    code_label = code_data['label'].values.tolist()
    text = text_data['text'].values.tolist()
    text_label = text_data['label'].values.tolist()
    for c, cl, t, tl in zip(code, code_label, text, text_label):
        if c != '' and t != '' and int(cl) == int(tl):
            examples.append(
                Example(c, t, int(cl))
            )
        else:
            break
    return examples

def mini_sample(examples, num):
    # 随机抽取num个数据
    example1 = []
    unique_numbers = random.sample(range(0, len(examples)), num)  # 修改范围根据需求调整
    for n in unique_numbers:
        for example_index, example in enumerate(examples):
            if example_index == n:
                example1.append(example)
    return example1

def text_to_feature(examples, code_tokenizer, text_tokenizer, stage = None):
    # 提取特征向量
    features = []
    for example_index, example in enumerate(examples):
        #CodeBERT部分
        # 使用tokenizer.tokenize()手动分词
        code_tokens = code_tokenizer.tokenize(example.code)[:510]
        code_tokens = [code_tokenizer.cls_token] + code_tokens + [code_tokenizer.sep_token]

        # 使用tokenizer.convert_tokens_to_ids()将分词后的文本转换为模型输入
        inputs_code_ids = code_tokenizer.convert_tokens_to_ids(code_tokens)
        inputs_code_masks = [1] * (len(code_tokens))
        code_padding_length = 512 - len(inputs_code_ids)
        inputs_code_ids += [code_tokenizer.pad_token_id] * code_padding_length
        inputs_code_masks += [0] * code_padding_length

        # RoBERTa部分
        # 使用tokenizer.tokenize()手动分词
        text_tokens = text_tokenizer.tokenize(example.text)[:510]
        text_tokens = [text_tokenizer.cls_token] + text_tokens + [text_tokenizer.sep_token]

        # 使用tokenizer.convert_tokens_to_ids()将分词后的文本转换为模型输入
        inputs_text_ids = text_tokenizer.convert_tokens_to_ids(text_tokens)
        inputs_text_masks = [1] * (len(text_tokens))
        text_padding_length = 512 - len(inputs_text_ids)
        inputs_text_ids += [text_tokenizer.pad_token_id] * text_padding_length
        inputs_text_masks += [0] * text_padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("id: {}".format(example_index))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in code_tokens]))
                logger.info("inputs_code_ids: {}".format(' '.join(map(str, inputs_code_ids))))
                logger.info("inputs_code_mask: {}".format(' '.join(map(str, inputs_code_masks))))
                logger.info("text_tokens: {}".format([x.replace('\u0120', '_') for x in text_tokens]))
                logger.info("inputs_text_ids: {}".format(' '.join(map(str, inputs_text_ids))))
                logger.info("inputs_text_mask: {}".format(' '.join(map(str, inputs_text_masks))))

        features.append(
            InputFeatures(
                inputs_code_ids,
                inputs_code_masks,
                inputs_text_ids,
                inputs_text_masks,
                example.label
            )
        )
    return features

def set_seed(seed=42):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)

def evaluate(eval_dataloader, model, device):
    start_time = time.time()
    total_correct = 0.0
    total_examples = 0.0
    all_pre = []
    all_labels = []
    model.eval()
    for bidx, batch in enumerate(eval_dataloader):
        inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask, inputs_label = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
        mlp_output = model(inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask)
        pred = torch.argmax(mlp_output, dim=1)
        all_labels += inputs_label.tolist()
        all_pre += pred.tolist()
        correct = torch.sum(pred == inputs_label)
        total_correct += correct.item()
        total_examples += int(mlp_output.size(0))
    acc = total_correct / total_examples
    f1 = f1_score(y_true=all_labels, y_pred=all_pre)
    rec = recall_score(y_true=all_labels, y_pred=all_pre)
    prec = precision_score(y_true=all_labels, y_pred=all_pre)
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        'acc': acc,
        'f1': f1,
        'rec': rec,
        'prec': prec,
        'execution_time': execution_time
    }

def main():
    epochs = 100
    batchsize = 4
    #set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    code_model = 'model/models--Salesforce--codet5p-110m-embedding/snapshots/d9f3a534af4252f04b10cd9f78e05037542d10f6'
    text_model = 'model/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    code_tokenizer = AutoTokenizer.from_pretrained(code_model, trust_remote_code=True)
    code_model = AutoModel.from_pretrained(code_model, trust_remote_code=True)
    text_tokenizer = RobertaTokenizer.from_pretrained(text_model)
    text_model = RobertaModel.from_pretrained(text_model)
    #model = Code_Note(code_model, text_model, 1536, 3072, 384)
    model = Code_Note(code_model, text_model, 768, 1536, 384)
    model.to(device)

    train_codefile = '../data/small_trvd/small_our_train.csv'
    train_textfile = '../data/small_trvd/ss_train.csv'
    eval_codefile = '../data/small_trvd/small_our_val.csv'
    eval_textfile = '../data/small_trvd/ss_val.csv'

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)

    #读取代码和注释(训练部分)
    examples = read_file(train_codefile, train_textfile)
    # examples = mini_sample(examples, 10)
    train_examples = text_to_feature(examples, code_tokenizer, text_tokenizer, 'train')
    all_inputs_code_ids = torch.tensor([f.inputs_code_ids for f in train_examples])
    all_inputs_code_masks = torch.tensor([f.inputs_code_masks for f in train_examples])
    all_inputs_text_ids = torch.tensor([f.inputs_text_ids for f in train_examples])
    all_inputs_text_masks = torch.tensor([f.inputs_text_masks for f in train_examples])
    all_inputs_labels = torch.tensor([f.label for f in train_examples])
    train_data = TensorDataset(all_inputs_code_ids, all_inputs_code_masks, all_inputs_text_ids, all_inputs_text_masks, all_inputs_labels)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)


    # 读取代码和注释(验证部分)
    eva_examples = read_file(eval_codefile, eval_textfile)
    eval_examples = text_to_feature(eva_examples, code_tokenizer, text_tokenizer, 'eval')
    all_evalinputs_code_ids = torch.tensor([f.inputs_code_ids for f in eval_examples])
    all_evalinputs_code_masks = torch.tensor([f.inputs_code_masks for f in eval_examples])
    all_evalinputs_text_ids = torch.tensor([f.inputs_text_ids for f in eval_examples])
    all_evalinputs_text_masks = torch.tensor([f.inputs_text_masks for f in eval_examples])
    all_evalinputs_labels = torch.tensor([f.label for f in eval_examples])
    eval_data = TensorDataset(all_evalinputs_code_ids, all_evalinputs_code_masks, all_evalinputs_text_ids, all_evalinputs_text_masks,
                               all_evalinputs_labels)
    eval_dataloader = DataLoader(eval_data, batch_size=batchsize, shuffle=True)
    
    # 学习率预热
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(epochs * len(train_dataloader)) * 0.1, num_training_steps=(epochs * len(train_dataloader)))

    best_metrics = {}
    best_epoch = 0
    with open("result.txt", "a") as f:
        # 迭代学习次数
        for epoch in range(epochs):
            train_total_lose = 0.0
            train_total_correct = 0.0
            train_total_examples = 0.0
            #训练部分
            model.train()
            loop = tqdm((train_dataloader), total=len(train_dataloader))
            for bidx, batch in enumerate(loop):
                inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask, inputs_label = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
                optimizer.zero_grad()
                mlp_output = model(inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask)
                # print(mlp_output)
                loss = criterion(mlp_output, inputs_label)
                loss.backward()
                optimizer.step()
                #scheduler.step()
                pred = torch.argmax(mlp_output, dim=1)
                # print(pred)
                train_total_lose += loss.item()
                correct = torch.sum(pred == inputs_label)
                train_total_correct += correct.item()
                train_total_examples += int(mlp_output.size(0))
                # print(total_correct,' ',total_examples)
                loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
                loop.set_postfix({'Train Loss': f'{train_total_lose/len(train_dataloader)}', 'Train ACC': f'{train_total_correct/train_total_examples}'})
            loop.close()

            #验证部分
            metrics = evaluate(eval_dataloader, model, device)
            eval_acc = metrics['acc']
            eval_f1 = metrics['f1']
            eval_rec = metrics['rec']
            eval_prec = metrics['prec']
            eval_time = metrics['execution_time']
            f.write(f'Epoch [{epoch + 1}/{epochs}] val_time: {eval_time}seconds    val_acc={eval_acc}, val_f1={eval_f1}, val_recall={eval_rec}, val_precision={eval_prec}'+'\n')
            print(f'val_time: {eval_time}seconds    val_acc={eval_acc}, val_f1={eval_f1}, val_recall={eval_rec}, val_precision={eval_prec}')
            if epoch == 0:
                best_metrics = metrics
                best_epoch = epoch
            else:
                if eval_acc >= best_metrics['acc']:
                    best_metrics = metrics
                    best_epoch = epoch
            best_acc = best_metrics['acc']
            best_f1 = best_metrics['f1']
            best_rec = best_metrics['rec']
            best_prec = best_metrics['prec']
            print('---best epoch---')
            print(f'best epcoch: {best_epoch+1}    acc={best_acc}, f1={best_f1}, recall={best_rec}, precision={best_prec}')



if __name__ == "__main__":
    main()
