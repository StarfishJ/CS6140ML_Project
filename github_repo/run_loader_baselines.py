"""Naive switch baselines for all 15 SwitchLingua pairs, read through streaming_loader.py.
Naive (current word): P(switch | pair, current word x_t), estimated on train; words seen fewer times are shrunk
toward their current-LID rate (WORD_PRIOR pseudo-counts); threshold tuned on each pair's validation split.
Uses only x_t: no punctuation after it, no earlier context.
Current LID: the coarser variant that only knows the language of x_t: P(switch | pair, LID of x_t) with add-one
smoothing from train; threshold tuned on each pair's validation split (and, for reference, one pooled threshold).
References: always stay (F1 = 0) and always switch (F1 = 2p / (1 + p) for switch rate p).
All scores are switch-class F1 on the test split of the fixed 70/10/20 split in split_manifest.json.
SDs are also reported per LID method, since label quality differs between script and lexicon LID.
Usage: python run_loader_baselines.py --workers 6 --out loader_baseline_results.json
       (all rows; --rows-per-pair N for a quick run; results do not depend on --workers)
"""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean,pstdev
from streaming_loader import DEFAULT_DATA_DIR,PAIRS,StreamingCSDataset

def metrics(true,pred):
    """Switch-class (positive = 1) precision, recall and F1, rounded to 4 places."""
    tp=sum(y==1 and p==1 for y,p in zip(true,pred));fp=sum(y==0 and p==1 for y,p in zip(true,pred));fn=sum(y==1 and p==0 for y,p in zip(true,pred))
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    return dict(n=len(true),positive=sum(true),precision=round(precision,4),recall=round(recall,4),f1=round(2*precision*recall/(precision+recall),4) if precision+recall else 0)

ORIGINAL4=['Arabic–English','Hindi–English','Korean–English','Russian–English']
WORD_PRIOR=5  # pseudo-counts pulling a rare word's switch rate toward its language's rate
SHARDS=4      # row shards per pair when --workers > 1

def collect(task):
    """Examples of one (pair file, shard) as (pair, split, current_lid, y_sw, y_dur, current word)."""
    fn,shard,data_dir,rows,cjk_seg=task
    if shard:
        import torch;torch.set_num_threads(1)  # one thread per process: the processes run in parallel instead
    return [(ex['pair'],ex['split'],ex['current_lid'],ex['y_sw'],ex['y_dur'],ex['window'][-1].lower())
            for ex in StreamingCSDataset(data_dir,[fn],split=None,rows_per_pair=rows,cjk_seg=cjk_seg,shard=shard)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    ap.add_argument('--rows-per-pair',type=int,default=None);ap.add_argument('--cjk-seg',choices=['word','char'],default='word')
    ap.add_argument('--out',type=Path,default=Path('loader_baseline_results.json'))
    ap.add_argument('--workers',type=int,default=1,help='parallel processes; results do not depend on it');a=ap.parse_args()
    data=defaultdict(lambda:defaultdict(list))  # pair -> split -> [(current_lid, y_sw, y_dur)]
    words=defaultdict(lambda:defaultdict(list))  # pair -> split -> [current word x_t, lowercased]
    if a.workers>1:
        import multiprocessing as mp
        tasks=[(fn,(i,SHARDS),a.data_dir,a.rows_per_pair,a.cjk_seg) for fn in PAIRS for i in range(SHARDS)]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            chunks=sorted(pool.imap_unordered(collect,tasks),key=lambda c:c[0][:2] if c else ('',''))
    else:
        chunks=[collect((fn,None,a.data_dir,a.rows_per_pair,a.cjk_seg)) for fn in PAIRS]
    for chunk in chunks:
        for pair,split,lid,y,dur,w in chunk:
            data[pair][split].append((lid,y,dur));words[pair][split].append(w)
    counts=defaultdict(Counter)
    for pair,d in data.items():
        for lid,y,_ in d['train']:counts[pair,lid][y]+=1
    def prob(pair,lid):
        c=counts[pair,lid];return (c[1]+1)/(c[1]+c[0]+2)
    grid=[i/100 for i in range(1,51)]
    def tune(val):return max(grid,key=lambda th:metrics([y for *_,y in val],[int(prob(p,l)>=th) for p,l,_ in val])['f1'])
    best=tune([(pair,lid,y) for pair,d in data.items() for lid,y,_ in d['val']])
    pair_best={pair:tune([(pair,lid,y) for lid,y,_ in d['val']]) for pair,d in data.items()}
    wcounts=defaultdict(Counter)
    for pair,d in data.items():
        for w,(_,y,_) in zip(words[pair]['train'],d['train']):wcounts[pair,w][y]+=1
    def wprob(pair,w,lid):
        c=wcounts[pair,w];return (c[1]+WORD_PRIOR*prob(pair,lid))/(sum(c.values())+WORD_PRIOR)
    wgrid=[i/100 for i in range(1,96)]
    def wpreds(pair,split,th):return [int(wprob(pair,w,l)>=th) for w,(l,_,_) in zip(words[pair][split],data[pair][split])]
    word_best={pair:max(wgrid,key=lambda th:metrics([y for _,y,_ in d['val']],wpreds(pair,'val',th))['f1']) for pair,d in data.items()}
    result={}
    for pair,d in data.items():
        test=d['test'];y=[x[1] for x in test]
        result[pair]={'always_stay':metrics(y,[0]*len(test)),
                      'always_switch':metrics(y,[1]*len(test)),  # reference: F1 = 2p/(1+p) for switch rate p
                      'lid_majority_0_50':metrics(y,[int(prob(pair,l)>=.5) for l,_,_ in test]),
                      'lid_threshold_val_tuned':metrics(y,[int(prob(pair,l)>=best) for l,_,_ in test]),
                      'pair_threshold':pair_best[pair],
                      'lid_threshold_pair_tuned':metrics(y,[int(prob(pair,l)>=pair_best[pair]) for l,_,_ in test]),
                      'word_pair_threshold':word_best[pair],
                      'word_threshold_pair_tuned':metrics(y,wpreds(pair,'test',word_best[pair])),
                      'switch_rate':round(sum(y)/len(y),4),
                      'split_positions':{s:len(d[s]) for s in ('train','val','test')},
                      'duration_labeled':sum(x[2]>=0 for x in test if x[1]==1),
                      'duration_unresolved':sum(x[2]<0 for x in test if x[1]==1)}
    f1={p:r['lid_threshold_val_tuned']['f1'] for p,r in result.items()}
    f1p={p:r['lid_threshold_pair_tuned']['f1'] for p,r in result.items()}
    f1w={p:r['word_threshold_pair_tuned']['f1'] for p,r in result.items()}
    method={pair:m for _,pair,m in PAIRS.values()}
    groups={g:[p for p in f1p if method[p]==g] for g in ('script','hindi','lexicon')}
    out={'setup':f'rows_per_pair={a.rows_per_pair or "all"}; fixed split from split_manifest.json (70/10/20, exact duplicates '
                 f'removed, Cantonese downsampled after splitting); word_lid.py labels; cjk_seg={a.cjk_seg}; counts only, no dataset text',
         'selected_validation_threshold':best,'pair_results':result,
         'tuned_f1_mean':round(mean(f1.values()),4),'universality_sd_tuned_f1':round(pstdev(f1.values()),4),
         'universality_sd_tuned_f1_original4':round(pstdev(f1[p] for p in ORIGINAL4),4),
         'pair_tuned_f1_mean':round(mean(f1p.values()),4),'universality_sd_pair_tuned_f1':round(pstdev(f1p.values()),4),
         'universality_sd_pair_tuned_f1_original4':round(pstdev(f1p[p] for p in ORIGINAL4),4),
         'word_f1_mean':round(mean(f1w.values()),4),'universality_sd_word_f1':round(pstdev(f1w.values()),4),
         'pair_tuned_by_lid_method':{g:{'pairs':ps,'mean_f1':round(mean(f1p[p] for p in ps),4),
                                        'sd_f1':round(pstdev(f1p[p] for p in ps),4)} for g,ps in groups.items() if ps},
         'limitations':['F1 measures weak labels, not gold LIDs: Unicode script (non-Latin), muril-hinglish-lid (romanized Hindi), '
                        'wordfreq comparison (Latin-script pairs; ambiguous words resolved from both sides)',
                        'Labels may read the whole sentence; the current-LID input feature uses only the prefix '
                        '(left-context rule for Latin-script pairs, the model on each prefix for romanized Hindi)',
                        'Chinese/Cantonese words come from jieba/PyCantonese; Japanese units are GiNZA bunsetsu '
                        '(content word + particles/auxiliaries, comparable to Korean eojeol); '
                        'Vietnamese syllables merged by pyvi',
                        'Only exact duplicates removed; near-duplicate utterances may still cross splits',
                        'Thresholds selected on validation splits only (pooled, or per pair)']}
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
