"""Streaming DataLoader for anticipatory code-switch prediction on SwitchLingua (all 15 pairs).
Reads CSV rows lazily, takes each source row's train/val/test split from split_manifest.json (make_split.py:
70/10/20, exact duplicates removed, rows sharing an utterance kept together, Cantonese downsampled), and yields
one example per word position t: the context window [x_{t-N+1}, ..., x_t] plus y_sw / y_dur for the
upcoming word x_{t+1}. Words after x_t are used only to build labels, never put in the window or input_ids.
Segmentation: letter-run words (combining marks such as Devanagari vowel signs kept); runs of Chinese /
Japanese / Cantonese characters (kanji, kana) are split by jieba / GiNZA bunsetsu (or Sudachi C, fugashi) / PyCantonese,
or into single characters with cjk_seg='char'; Vietnamese syllables are merged into words by pyvi (also
cjk_seg='word'). Labels are per word; with a tokenizer, the prediction point is the last subword of x_t.
Word LID: word_lid.py (script, muril-hinglish-lid for romanized Hindi, wordfreq for Latin-script pairs);
lid_fn(words, lang, pair) -> list of labels overrides it.
Usage: python streaming_loader.py --demo [--tokenizer xlm-roberta-base] [--cjk-seg char]
       python streaming_loader.py --stats --out loader_stats.json   (data: $SWITCHLINGUA_DIR or data/SwitchLingua_text)
Requires: wordfreq; jieba, pycantonese, ginza, ja_ginza (or sudachipy / fugashi), pyvi (only for cjk_seg='word');
transformers (only with a tokenizer).
"""
import argparse,ast,csv,json,os,re,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from word_lid import causal_lid_sequence,confident_lid,lexicon_lid,lid_sequence
try:
    from torch.utils.data import IterableDataset as _Base  # lets torch.utils.data.DataLoader wrap the dataset
except ImportError:
    _Base=object

# file -> (other language, pair name, LID method): 'script' = Unicode script; 'hindi' = script + romanized-Hindi
# lexicon; 'lexicon' = wordfreq comparison of the two languages (see word_lid.py).
PAIRS={'Arabic_eng.csv':('Arabic','Arabic–English','script'),
       'Hindi_eng.csv':('Hindi','Hindi–English','hindi'),
       'Korean_eng.csv':('Korean','Korean–English','script'),
       'Russian_eng.csv':('Russian','Russian–English','script'),
       'Chinese_eng.csv':('Chinese','Chinese–English','script'),
       'Can_Eng.csv':('Cantonese','Cantonese–English','script'),
       'Japanese_eng.csv':('Japanese','Japanese–English','script'),
       'Spanish_eng.csv':('Spanish','Spanish–English','lexicon'),
       'French_eng.csv':('French','French–English','lexicon'),
       'German_Eng.csv':('German','German–English','lexicon'),
       'Italian_eng.csv':('Italian','Italian–English','lexicon'),
       'Malay_eng.csv':('Malay','Malay–English','lexicon'),
       'Philippines_eng.csv':('Filipino','Filipino–English','lexicon'),
       'Vietnamese_eng.csv':('Vietnamese','Vietnamese–English','lexicon'),
       'German_French.csv':('German','German–French','lexicon')}  # no English side: labels are German / French
def _default_data_dir():
    """$SWITCHLINGUA_DIR if set, else data/SwitchLingua_text next to this file, else one level up."""
    if os.environ.get('SWITCHLINGUA_DIR'):return Path(os.environ['SWITCHLINGUA_DIR'])
    here=Path(__file__).resolve().parent
    return next((p for p in (here/'data/SwitchLingua_text',here.parent/'data/SwitchLingua_text') if p.exists()),here/'data/SwitchLingua_text')
DEFAULT_DATA_DIR=_default_data_dir()

def utterances(cell):
    """The data_generation_result cell is a serialized Python list of utterances (or a single string)."""
    try:
        obj=ast.literal_eval(cell)
        if isinstance(obj,list):return [x for x in obj if isinstance(x,str)]
        if isinstance(obj,str):return [obj]
    except (SyntaxError,ValueError):pass
    return [cell] if isinstance(cell,str) and cell.strip() else []
DEFAULT_MANIFEST=Path(__file__).resolve().parent/'split_manifest.json'
SPLIT_NAMES={'T':'train','V':'val','E':'test'}  # 'D' duplicate and 'U' downsampled rows are skipped

CJK=r'々-〇぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ\U00020000-\U0003134f'
# \w excludes combining marks (Mn/Mc/Me, e.g. Devanagari vowel signs, virama), so add them or words get split
MARKS=re.escape(''.join(chr(c) for c in range(0x300,0x10000) if unicodedata.category(chr(c)).startswith('M')))
WORD=rf"[^\W_{CJK}](?:[^\W_{CJK}]|[{MARKS}])*"
TOKEN=re.compile(rf"(?P<cjk>[{CJK}]+)|{WORD}(?:['’\-]{WORD})*")

JA_SEG='ginza'  # GiNZA bunsetsu; or 'sudachi-C' (compound nouns kept, particles split) / 'fugashi' (short units)
_SEG={}
def cjk_segmenter(lang):
    """Word segmenter for a run of CJK characters (no spaces), loaded on first use."""
    if lang not in _SEG:
        if lang=='Chinese':
            import jieba;jieba.setLogLevel(60);_SEG[lang]=jieba.lcut
        elif lang=='Cantonese':
            import pycantonese;_SEG[lang]=pycantonese.segment
        elif lang=='Japanese' and JA_SEG=='ginza':  # bunsetsu = content word + its particles/auxiliaries, like Korean eojeol
            import spacy,ginza;nlp=spacy.load('ja_ginza');_SEG[lang]=lambda s:[b.text for b in ginza.bunsetu_spans(nlp(s))]
        elif lang=='Japanese' and JA_SEG=='fugashi':
            import fugashi;tagger=fugashi.Tagger();_SEG[lang]=lambda s:[w.surface for w in tagger(s)]
        elif lang=='Japanese':  # Sudachi split mode C keeps compound nouns whole (東京都庁, 新型コロナウイルス感染症)
            from sudachipy import Dictionary,SplitMode
            sudachi=Dictionary(dict='core').create();_SEG[lang]=lambda s:[m.surface() for m in sudachi.tokenize(s,SplitMode.C)]
        else:_SEG[lang]=list  # e.g. hanja inside Korean text: one token per character
    return _SEG[lang]

def vi_merge(text,spans):
    """Merge whitespace-separated syllables that pyvi joins into one word (e.g. 'học_sinh').
    pyvi tokens are matched to our spans in order; anything that does not line up stays unmerged."""
    if 'vi' not in _SEG:
        from pyvi import ViTokenizer;_SEG['vi']=ViTokenizer.tokenize
    words=[text[a:b].lower() for a,b in spans];out=[];i=0
    for tok in _SEG['vi'](text).split():
        parts=tok.lower().split('_')
        if words[i:i+len(parts)]!=parts:continue
        k=len(parts)
        # pyvi sometimes joins English words ('The_media'): skip the merge only when every syllable is English,
        # so compounds containing an English-looking syllable ('an toàn') still merge
        if k>1 and all(text[spans[j][1]:spans[j+1][0]].isspace() for j in range(i,i+k-1)) \
                and any(lexicon_lid(p,'Vietnamese–English',None)!='en' for p in parts):
            out.append((spans[i][0],spans[i+k-1][1]))
        else:out.extend(spans[i:i+k])
        i+=k
    return out+spans[i:]

def split_scripts(text,a,b):
    """Cut a token where its letters change between Latin and another script ('meeting에서' -> meeting + 에서)."""
    cuts=[a];prev=None
    for i in range(a,b):
        if not text[i].isalpha():continue  # combining marks / hyphens stay with the left piece
        latin='LATIN' in unicodedata.name(text[i],'')
        if prev is not None and latin!=prev:cuts.append(i)
        prev=latin
    out=[]
    for s,e in zip(cuts,cuts[1:]+[b]):
        while s<e and text[s] in "'’-":s+=1
        while e>s and text[e-1] in "'’-":e-=1
        if s<e:out.append((s,e))
    return out

def split_hyphen(text,a,b,pair,method,data_dir):
    """Split a hyphenated Latin-script word whose parts are confidently in different languages
    ('nag-meeting', 'Meeting-Raum'); same-language compounds ('well-known', 'anak-anak') stay whole."""
    if '-' not in text[a:b] or method not in ('lexicon','hindi'):return [(a,b)]
    parts=[];s=a
    for p in text[a:b].split('-'):
        if p:parts.append((s,s+len(p)))
        s+=len(p)+1
    labels={confident_lid(text[x:y],pair,method,data_dir) for x,y in parts}-{None}
    return parts if len(labels)>1 else [(a,b)]

def tokenize(text,lang,cjk_seg='word',pair=None,method=None,data_dir=DEFAULT_DATA_DIR):
    """Word spans. A CJK run is split into words (cjk_seg='word') or single characters ('char');
    Vietnamese syllables are merged into words when cjk_seg='word'. Words mixing two languages are split:
    at a Latin / non-Latin script boundary, or at a hyphen (needs pair and method)."""
    spans=[]
    for m in TOKEN.finditer(text):
        a,b=m.span();run=m.group('cjk')
        if run is None:
            for s,e in split_scripts(text,a,b):spans.extend(split_hyphen(text,s,e,pair,method,data_dir))
            continue
        pieces=cjk_segmenter(lang)(run) if cjk_seg=='word' else list(run)
        if ''.join(pieces)!=run:pieces=list(run)  # segmenter altered the text: fall back to characters
        for p in pieces:spans.append((a,a+len(p)));a+=len(p)
    if lang=='Vietnamese' and cjk_seg=='word':spans=vi_merge(text,spans)
    return spans

SENT_END=re.compile(r'[.!?…:;"“”。！？]\s*$')
def sentence_starts(text,spans):
    """True for the first word and for words right after sentence-final punctuation."""
    return [i==0 or bool(SENT_END.search(text[spans[i-1][1]:spans[i][0]])) for i in range(len(spans))]

def dur_bin(n):return 0 if n<=2 else 1 if n<=6 else 2

def targets(ids,t):
    """(y_sw, y_dur) for the word after position t, or None if either LID is unknown."""
    if ids[t] is None or ids[t+1] is None:return None
    if ids[t]==ids[t+1]:return 0,-1
    j=t+1
    while j<len(ids) and ids[j]==ids[t+1]:j+=1
    if j<len(ids) and ids[j] is None:return 1,-1  # new span ends at an unknown word: length cannot be certified
    return 1,dur_bin(j-t-1)

_MANIFESTS={}
def row_splits(fn,manifest=DEFAULT_MANIFEST):
    """Per-row split codes for one CSV file from the fixed manifest (run make_split.py to create it)."""
    key=str(manifest)
    if key not in _MANIFESTS:
        if not Path(manifest).exists():raise FileNotFoundError(f'{manifest} not found: run make_split.py first')
        _MANIFESTS[key]=json.loads(Path(manifest).read_text(encoding='utf-8'))['files']
    return _MANIFESTS[key][fn]

def encode_context(context,tokenizer,max_len):
    """Subword ids for the window text; label_pos = last non-special subword, i.e. the last piece of x_t."""
    enc=tokenizer(context,truncation=True,max_length=max_len,return_special_tokens_mask=True)
    pos=max(i for i,sp in enumerate(enc['special_tokens_mask']) if not sp)
    return enc['input_ids'],pos

def examples_from_text(text,lang,pair,method='script',window=32,tokenizer=None,max_len=128,lid_fn=None,cjk_seg='word',
                       data_dir=DEFAULT_DATA_DIR,causal_feature=True):
    """Labels (y_sw, y_dur) come from label LIDs, which may read the whole sentence (romanized Hindi);
    current_lid is the input feature, computed from the prefix only (causal_feature=False reuses label LIDs)."""
    spans=tokenize(text,lang,cjk_seg,pair,method,data_dir)
    words=[text[a:b] for a,b in spans]
    starts=sentence_starts(text,spans)
    ids=lid_fn(words,lang,pair) if lid_fn else lid_sequence(words,lang,pair,method,data_dir,starts)
    feats=causal_lid_sequence(words,lang,pair,method,ids,starts) if causal_feature and not lid_fn else ids
    for t in range(len(words)-1):
        y=targets(ids,t)
        if y is None:continue
        s=max(0,t-window+1)
        ex=dict(pair=pair,t=t,window=words[s:t+1],current_lid=feats[t],y_sw=y[0],y_dur=y[1])
        if tokenizer is not None:
            # slice the raw text up to the end of x_t, so the tokenizer never sees x_{t+1}
            ex['input_ids'],ex['label_pos']=encode_context(text[spans[s][0]:spans[t][1]],tokenizer,max_len)
        yield ex

class StreamingCSDataset(_Base):
    """Iterable over prefix-window examples. split=None yields all splits (each example carries 'split').
    rows_per_pair only reads the first N CSV rows per file (quick runs); duplicates and downsampled rows are skipped.
    lid_fn(words, lang, pair) -> list of labels (None = unknown) overrides word_lid.lid_sequence.
    shard=(i, k) keeps only rows with row_id % k == i, so k processes (or torch DataLoader workers) can split the
    stream; without it, each worker would replay the full stream."""
    def __init__(self,data_dir,pairs=None,split='train',window=32,rows_per_pair=None,tokenizer=None,max_len=128,lid_fn=None,
                 cjk_seg='word',manifest=DEFAULT_MANIFEST,causal_feature=True,shard=None):
        self.data_dir=Path(data_dir);self.pairs=list(pairs or PAIRS);self.split=split;self.window=window
        self.rows_per_pair=rows_per_pair;self.tokenizer=tokenizer;self.max_len=max_len;self.lid_fn=lid_fn;self.cjk_seg=cjk_seg
        self.manifest=manifest;self.causal_feature=causal_feature;self.shard=shard
        if tokenizer is not None:tokenizer.truncation_side='left'  # keep x_t when the window exceeds max_len
        for fn in self.pairs:
            if fn not in PAIRS:raise ValueError(f'unknown pair file {fn}')
    def __iter__(self):
        for fn in self.pairs:
            lang,pair,method=PAIRS[fn];codes=row_splits(fn,self.manifest)
            with open(self.data_dir/fn,encoding='utf-8-sig',newline='') as f:
                for row_id,row in enumerate(csv.DictReader(f)):
                    if self.rows_per_pair is not None and row_id>=self.rows_per_pair:break
                    if self.shard and row_id%self.shard[1]!=self.shard[0]:continue
                    split=SPLIT_NAMES.get(codes[row_id])
                    if split is None or (self.split is not None and split!=self.split):continue
                    for u_id,text in enumerate(utterances(row['data_generation_result'])):
                        for ex in examples_from_text(text,lang,pair,method,self.window,self.tokenizer,self.max_len,self.lid_fn,
                                                     self.cjk_seg,self.data_dir,self.causal_feature):
                            ex.update(split=split,row=row_id,utt=u_id);yield ex

def collate(batch,pad_id):
    """Right-pad input_ids; label_pos indexes stay valid."""
    L=max(len(e['input_ids']) for e in batch)
    return dict(input_ids=[e['input_ids']+[pad_id]*(L-len(e['input_ids'])) for e in batch],
                attention_mask=[[1]*len(e['input_ids'])+[0]*(L-len(e['input_ids'])) for e in batch],
                label_pos=[e['label_pos'] for e in batch],y_sw=[e['y_sw'] for e in batch],y_dur=[e['y_dur'] for e in batch])

def batches(dataset,batch_size,pad_id):
    buf=[]
    for ex in dataset:
        buf.append(ex)
        if len(buf)==batch_size:yield collate(buf,pad_id);buf=[]
    if buf:yield collate(buf,pad_id)

DEMO=[  # made-up sentences, not from SwitchLingua
    ('I want to 먹어요 정말 맛있는 김치찌개 today','Korean','Korean–English'),
    ('오늘 presentation에서 질문을 받았어요','Korean','Korean–English'),
    ('Nag-meeting kami kahapon about the new well-known project','Filipino','Filipino–English'),
    ('我们的 meeting 改到明天下午','Chinese','Chinese–English'),
    ('今日の presentation はすごく良かった','Japanese','Japanese–English'),
    ('我今日好 busy 呀','Cantonese','Cantonese–English'),
    ('मुझे यह movie बहुत पसंद आई','Hindi','Hindi–English'),
    ('mujhe yeh movie bahut pasand aayi','Hindi','Hindi–English'),
    ('Mañana tengo un meeting con mi jefe','Spanish','Spanish–English'),
    ('Hôm nay học sinh đi meeting với giáo viên','Vietnamese','Vietnamese–English'),
    ('Ich habe heute keine Zeit pour le déjeuner','German','German–French')]

def demo(tokenizer_name,window,cjk_seg):
    tok=None
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(tokenizer_name);tok.truncation_side='left'
    method={pair:m for _,pair,m in PAIRS.values()}
    for text,lang,pair in DEMO:
        print(f'\n== {pair}: {text}')
        words=[text[a:b] for a,b in tokenize(text,lang,cjk_seg,pair,method[pair])]
        for ex in examples_from_text(text,lang,pair,method[pair],window=window,tokenizer=tok,cjk_seg=cjk_seg):
            line=f"t={ex['t']:<2} window={ex['window']}  next={words[ex['t']+1]!r:<14} y_sw={ex['y_sw']} y_dur={ex['y_dur']:>2}"
            if tok:line+=f"  predict at {tok.convert_ids_to_tokens(ex['input_ids'][ex['label_pos']])!r} ({len(ex['input_ids'])} ids)"
            print(line)

def stats(data_dir,rows_per_pair,window,cjk_seg):
    """Label counts per pair/split (counts only, no dataset text)."""
    out={}
    for fn,(lang,pair,method) in PAIRS.items():
        c=defaultdict(Counter)
        for ex in StreamingCSDataset(data_dir,[fn],split=None,window=window,rows_per_pair=rows_per_pair,cjk_seg=cjk_seg,
                                     causal_feature=False):  # label counts only
            s=c[ex['split']];s['positions']+=1;s['switches']+=ex['y_sw']
            if ex['y_sw']:s['dur_%d'%ex['y_dur'] if ex['y_dur']>=0 else 'dur_unresolved']+=1
        out[pair]={sp:dict(positions=v['positions'],switches=v['switches'],switch_rate=round(v['switches']/v['positions'],4),
                           duration_bins=[v['dur_0'],v['dur_1'],v['dur_2']],duration_unresolved=v['dur_unresolved'])
                   for sp,v in sorted(c.items())}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    ap.add_argument('--rows-per-pair',type=int,default=None);ap.add_argument('--window',type=int,default=32)
    ap.add_argument('--tokenizer');ap.add_argument('--out',type=Path);ap.add_argument('--cjk-seg',choices=['word','char'],default='word')
    ap.add_argument('--demo',action='store_true');ap.add_argument('--stats',action='store_true');a=ap.parse_args()
    if a.demo:return demo(a.tokenizer,min(a.window,4),a.cjk_seg)
    if not a.stats:ap.error('use --demo or --stats')
    out=dict(setup=f'rows_per_pair={a.rows_per_pair or "all"}; split from split_manifest.json (70/10/20, deduplicated); '
                   f'window={a.window} words; cjk_seg={a.cjk_seg}',
             pairs=stats(a.data_dir,a.rows_per_pair,a.window,a.cjk_seg))
    text=json.dumps(out,indent=2,ensure_ascii=False)
    if a.out:a.out.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()
