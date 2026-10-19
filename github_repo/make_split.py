"""Fixed train/val/test split (70/10/20) for all 15 SwitchLingua pairs, saved as split_manifest.json.
- Exact duplicate rows (same normalized utterance list) keep their first occurrence; later copies are 'D'.
- Rows sharing any normalized utterance of >= MIN_WORDS words are grouped (union-find) BEFORE splitting, and a whole
  group goes to one split, so no such utterance appears in two splits. Chinese/Japanese/Cantonese have no spaces,
  so their words are estimated as two characters each (n_words).
- Groups are shuffled with a per-file fixed seed and cut at 70% / 80% of rows: 'T' train, 'V' val, 'E' test.
- Cantonese–English is downsampled after splitting (each split keeps its share) to the median size of the other
  pairs; dropped rows are 'U'. The test split is therefore fixed before any sampling.
Manifest: {"files": {csv file: one code per CSV row}, settings, counts}. Counts only, no dataset text.
Usage: python make_split.py --out split_manifest.json   (data: $SWITCHLINGUA_DIR or data/SwitchLingua_text)
"""
import argparse,csv,json,random,re,unicodedata
from collections import Counter
from pathlib import Path
from statistics import median
from streaming_loader import PAIRS,DEFAULT_DATA_DIR,utterances

SEED=42;RATIOS=(0.7,0.1,0.2);MIN_WORDS=3;DOWNSAMPLE='Can_Eng.csv'

def norm(u):return re.sub(r'[\W_]+',' ',unicodedata.normalize('NFKC',u).lower()).strip()

CJK_CHAR=re.compile('[぀-ヿ㐀-䶿一-鿿豈-﫿]')
def n_words(u):
    """Approximate word count: spaces separate words, except in Chinese/Japanese/Cantonese text, which has
    no spaces, so every two CJK characters count as one word."""
    return len(CJK_CHAR.sub(' ',u).split())+len(CJK_CHAR.findall(u))/2

def split_file(path,rng):
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=[tuple(norm(u) for u in utterances(r['data_generation_result'])) for r in csv.DictReader(f)]
    codes=['']*len(rows);first={};parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    owner={}
    for i,us in enumerate(rows):
        if us in first:codes[i]='D';continue  # exact duplicate row
        first[us]=i
        for u in us:
            if n_words(u)<MIN_WORDS:continue
            if u in owner:parent[find(i)]=find(owner[u])
            else:owner[u]=i
    groups={}
    for i in range(len(rows)):
        if codes[i]!='D':groups.setdefault(find(i),[]).append(i)
    order=sorted(groups.values());rng.shuffle(order)
    kept=sum(map(len,order));cut1,cut2=RATIOS[0]*kept,(RATIOS[0]+RATIOS[1])*kept;seen=0
    for g in order:
        code='T' if seen<cut1 else 'V' if seen<cut2 else 'E'
        for i in g:codes[i]=code
        seen+=len(g)
    return codes,max(map(len,order))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'split_manifest.json');a=ap.parse_args()
    files={};largest={}
    # one generator per file, seeded by (SEED, file name): changing one pair never shifts another pair's split
    rngs={fn:random.Random(f'{SEED}:{fn}') for fn in PAIRS}
    for fn in PAIRS:files[fn],largest[fn]=split_file(a.data_dir/fn,rngs[fn])
    target=round(median(sum(c!='D' for c in codes) for fn,codes in files.items() if fn!=DOWNSAMPLE))
    codes=files[DOWNSAMPLE];kept=sum(c!='D' for c in codes)
    for s in 'TVE':  # sample within each split so the split shares stay 70/10/20
        idx=[i for i,c in enumerate(codes) if c==s]
        keep=set(rngs[DOWNSAMPLE].sample(idx,round(len(idx)*target/kept)))
        for i in idx:
            if i not in keep:codes[i]='U'
    counts={PAIRS[fn][1]:dict(Counter(c)) for fn,c in files.items()}
    out=dict(setup=f'seed={SEED}; ratios train/val/test={RATIOS}; exact-duplicate rows dropped (D); rows sharing a normalized '
                   f'utterance of >={MIN_WORDS} words kept in one split; {DOWNSAMPLE} downsampled after splitting to {target} rows (U)',
             codes='T=train V=val E=test D=duplicate row U=unused (downsampled)',
             counts=counts,largest_group={PAIRS[fn][1]:largest[fn] for fn in files},files={fn:''.join(c) for fn,c in files.items()})
    a.out.write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
    for p,c in counts.items():
        n=sum(c.get(k,0) for k in 'TVE');print(f"{p:20} T {c.get('T',0):>6} V {c.get('V',0):>5} E {c.get('E',0):>5} ({c.get('E',0)/n:.1%} test) "
                                             f"D {c.get('D',0):>5} U {c.get('U',0):>6} largest group {out['largest_group'][p]}")
if __name__=='__main__':main()
