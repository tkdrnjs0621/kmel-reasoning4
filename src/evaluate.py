'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from datasets import Dataset

def evaluate(text_model, dataset_path, text_trunc_length, out_column, reasoning):
    outputs = []

    # with open(osp.join(input_file), encoding='utf8') as f:
    #     reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    #     for n, line in enumerate(reader):
    #         out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
    #         outputs.append((line['SMILES'], line['ground truth'], out_tmp))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
    dataset = Dataset.from_json(dataset_path)
    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []
    print(reasoning)
    k=0
    for d in tqdm(dataset):

        gt = d['description']
        out = d[out_column]
        if(reasoning):
            out = out.lower().split('final description:')[-1]
            # delimiter = 'final description:'
            # lower_out = out.lower()
            # index = lower_out.rfind(delimiter)

            # if index != -1:
            #     out = out[index + len(delimiter):]
            # else:
            #     out = out

        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, d in enumerate(dataset):

        gt = d['description']
        out = d[out_column]

        if(reasoning):
            out = out.lower().split('final description:')[-1]
            # delimiter = 'final description:'
            # lower_out = out.lower()
            # index = lower_out.rfind(delimiter)

            # if index != -1:
            #     out = out[index + len(delimiter):]
            # else:
            #     out = out
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    # print('rouge1:', rouge_1*100)
    # print('rouge2:', rouge_2*100)
    # print('rougeL:', rouge_l*100)
    
    # print('BLEU-2 score:', round(bleu2*100,1))
    # print('BLEU-4 score:', round(bleu4*100,1))
    # print('Meteor score:', _meteor_score)
    # print("BLEU-2\tBLUE-4\trouge1\trouge2\trouge\tMETEORL")
    # print(round(blue2*100,1),round(blue4*100,1),round(rouge1*100,1),round(rouge2*100,1),round(rougeL*100,1),round(_meteor_score*100,1))
    print(f"{'BLEU-2':>8} {'BLEU-4':>8} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'METEOR':>8}")
    print(f"{bleu2*100:8.1f} {bleu4*100:8.1f} {rouge_1*100:8.1f} {rouge_2*100:8.1f} {rouge_l*100:8.1f} {_meteor_score*100:8.1f}")

    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model tokenizer.')
    parser.add_argument('--dataset_path', type=str, default='', help='path where test generations are saved')
    parser.add_argument('--out_column', type=str, default='prediction', help='path where test generations are saved')
    parser.add_argument('--reasoning', action='store_true', help='path where test generations are saved')
    parser.add_argument('--text_trunc_length', type=str, default=512, help='tokenizer maximum length')
    args = parser.parse_args()
    evaluate(args.text_model, args.dataset_path, args.text_trunc_length, args.out_column,args.reasoning)