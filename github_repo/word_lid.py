"""Word-level language ID for all 15 SwitchLingua pairs (weak labels: the dataset has no gold word LIDs).
- Non-Latin pairs (Arabic, Korean, Russian, Chinese, Cantonese, Japanese): Unicode script.
- Hindi: utterances containing Devanagari use script (Devanagari = Hindi, Latin = English). Fully romanized
  utterances are tagged by PhysicsWallahAI/muril-hinglish-lid (MuRIL fine-tuned on LinCE Hindi-English), which
  reads the whole sentence; it is not used on Devanagari utterances, where it mislabels some English words.
  Labels may use the whole sentence; the baseline input feature uses causal_lid_sequence (prefix only).
- Latin-script pairs: wordfreq frequency of the word in each of the two languages; a clear winner (>= MARGIN
  Zipf) decides, otherwise letters used by only one side (e.g. ñ, ß, ơ) decide. Labels resolve the remaining
  ambiguous words from decided words on both sides (lexicon_full_sequence); the input feature lets them
  inherit the previous word's language, so it never depends on x_{t+1}.
  A capitalized 'I' inside a sentence is the English pronoun (the Italian article 'i' is lowercase there).
- Multi-syllable Vietnamese words (merged by the loader) take their syllables' language if all agree.
- Numbers (no letters) are language-neutral and inherit the previous word's language (None at utterance start).
Returns 'en' for English and the language name otherwise; None = unknown (position excluded from labels).
"""
import unicodedata
from functools import lru_cache
from wordfreq import zipf_frequency

HINGLISH_MODEL='PhysicsWallahAI/muril-hinglish-lid'

MARGIN=1.0
# The Filipino and Malay frequency lists contain many English loans (half / a third of common English words
# score within 1.0 Zipf of English), so a smaller margin toward English keeps the first English word after a
# switch from inheriting the previous language. Other lists are cleaner, and a small margin there turns shared
# words (German 'Nation') into fake switches. Tuned on wordfreq top-3000 lists.
EN_MARGIN={'Filipino':0.5,'Malay':0.5}
# pair -> ((label, wordfreq codes), (label, wordfreq codes))
LEXICON={'Spanish–English':(('Spanish',['es']),('en',['en'])),
         'French–English':(('French',['fr']),('en',['en'])),
         'German–English':(('German',['de']),('en',['en'])),
         'Italian–English':(('Italian',['it']),('en',['en'])),
         'Malay–English':(('Malay',['ms','id']),('en',['en'])),  # Malay text often shares Indonesian spellings
         'Filipino–English':(('Filipino',['fil']),('en',['en'])),
         'Vietnamese–English':(('Vietnamese',['vi']),('en',['en'])),
         'German–French':(('German',['de']),('French',['fr']))}
# letters a side uses but the other does not (English uses none of them)
ONLY={'Spanish':set('ñáéíóúü'),'French':set('éèêëàâîïôùûüÿçœæ'),'German':set('äöüß'),'Italian':set('àèéìíòóù'),
      'Vietnamese':set('ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ'),
      'Malay':set(),'Filipino':set('ñ'),'en':set()}
SCRIPT_NAME={'Arabic':('ARABIC',),'Hindi':('DEVANAGARI',),'Korean':('HANGUL',),'Russian':('CYRILLIC',),
             'Chinese':('CJK','IDEOGRAPHIC'),'Cantonese':('CJK','IDEOGRAPHIC'),'Japanese':('CJK','IDEOGRAPHIC','HIRAGANA','KATAKANA')}

def char_tags(word,lang):
    tags=set()
    for c in word:
        if not c.isalpha():continue
        n=unicodedata.name(c,'')
        tags.add('en' if 'LATIN' in n else lang if any(s in n for s in SCRIPT_NAME[lang]) else '?')
    return tags

def script_lid(word,lang):
    tags=char_tags(word,lang)
    return next(iter(tags)) if len(tags)==1 and '?' not in tags else None

@lru_cache(maxsize=None)
def zipf(word,codes):
    return max(zipf_frequency(word,c) for c in codes)

def lexicon_lid(word,pair,prev,strict=False):
    """strict=True: return None instead of guessing when the word alone does not decide it."""
    (la,ca),(lb,cb)=LEXICON[pair]
    w=word.lower().replace('’',"'")  # wordfreq lists use the ASCII apostrophe (j'ai, don't)
    if not any(c.isalpha() for c in w) or any(c.isalpha() and 'LATIN' not in unicodedata.name(c,'') for c in w):return None
    za,zb=zipf(w,tuple(ca)),zipf(w,tuple(cb))
    if za-zb>=MARGIN:return la
    if zb-za>=(EN_MARGIN.get(la,MARGIN) if lb=='en' else MARGIN):return lb
    oa,ob=bool(set(w)&ONLY[la]-ONLY[lb]),bool(set(w)&ONLY[lb]-ONLY[la])
    if oa!=ob:return la if oa else lb
    if prev is not None:return prev
    if strict:return None
    return None if za==zb==0 else la if za>=zb else lb

_HLID={}
def _hinglish():
    """Load the Hinglish LID model once (from the local cache when available)."""
    if not _HLID:
        import torch
        from transformers import AutoTokenizer,AutoModelForTokenClassification
        try:
            tok=AutoTokenizer.from_pretrained(HINGLISH_MODEL,local_files_only=True)
            model=AutoModelForTokenClassification.from_pretrained(HINGLISH_MODEL,local_files_only=True)
        except OSError:
            tok=AutoTokenizer.from_pretrained(HINGLISH_MODEL);model=AutoModelForTokenClassification.from_pretrained(HINGLISH_MODEL)
        names={'HIN':'Hindi','ENG':'en'}
        _HLID.update(torch=torch,tok=tok,model=model.eval(),labels={int(i):names[l] for i,l in model.config.id2label.items()})
    return _HLID

def hinglish_tags(batch):
    """'Hindi'/'en' for every word of each word list in batch, read off the word's first subword (the model
    card's recipe); None for words cut off by truncation."""
    h=_hinglish()
    enc=h['tok'](batch,is_split_into_words=True,truncation=True,max_length=256,padding=True,return_tensors='pt')
    with h['torch'].no_grad():pred=h['model'](**enc).logits.argmax(-1).tolist()
    out=[]
    for b,words in enumerate(batch):
        labels=[None]*len(words);prev=None
        for pos,wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid!=prev:labels[wid]=h['labels'][pred[b][pos]]
            prev=wid
        out.append(labels)
    return out

def romanized_hindi(words,method):
    return method=='hindi' and bool(words) and not any(script_lid(w,'Hindi')=='Hindi' for w in words)

def confident_lid(word,pair,method,data_dir=None):
    """Label from the word alone (no context), or None when it is ambiguous; used to split mixed hyphen words."""
    if method=='hindi':
        l=script_lid(word,'Hindi');return hinglish_tags([[word]])[0][0] if l=='en' else l
    return lexicon_lid(word,pair,None,strict=True)

def weak_lid(word,pair):
    """The language whose list has the word more often, however small the gap; None if neither has it."""
    (la,ca),(lb,cb)=LEXICON[pair];w=word.lower().replace('’',"'")
    za,zb=zipf(w,tuple(ca)),zipf(w,tuple(cb))
    return None if za==zb==0 else la if za>=zb else lb

def lexicon_full_sequence(words,pair,initial=None):
    """Label LIDs for Latin-script pairs, reading the whole sentence. Words the frequency lists decide stay as
    they are; an ambiguous word takes the language of the nearest decided words on both sides when they agree
    (or the one side that exists), and at a switch boundary, where they differ, the more frequent language."""
    la=LEXICON[pair][0][0];en_pair=LEXICON[pair][1][0]=='en';conf=[]
    for i,w in enumerate(words):
        start=initial[i] if initial is not None else i==0
        if not any(c.isalpha() for c in w):conf.append('#')  # number: resolved after its neighbours
        elif any(c.isalpha() and 'LATIN' not in unicodedata.name(c,'') for c in w):conf.append('x')  # other script
        elif en_pair and not start and w=='I':conf.append('en')
        elif ' ' in w:conf.append(la)  # pyvi compound: Vietnamese
        else:conf.append(lexicon_lid(w,pair,None,strict=True))
    decided=lambda c:c not in (None,'#','x')
    out=[]
    for i,c in enumerate(conf):
        if c is None:
            left=next((conf[j] for j in range(i-1,-1,-1) if decided(conf[j])),None)
            right=next((conf[j] for j in range(i+1,len(conf)) if decided(conf[j])),None)
            c=left if left==right else right if left is None else left if right is None else weak_lid(words[i],pair)
            if c is None:c=weak_lid(words[i],pair)
        out.append(c)
    ids=[];prev=None
    for c in out:
        l=prev if c=='#' else None if c=='x' else c  # numbers inherit the previous language
        ids.append(l)
        if l is not None:prev=l
    return ids

def lid_sequence(words,lang,pair,method,data_dir=None,initial=None,context='full'):
    """LIDs per word. context='full' gives label LIDs (may read the whole sentence); context='left' uses only the
    current and earlier words, which is how Latin-script pairs build the input feature.
    initial[i]: word i starts a sentence (default: only the first word); a sentence-initial 'I' may be an Italian
    article, a mid-sentence one is the English pronoun."""
    if method=='lexicon' and context=='full':return lexicon_full_sequence(words,pair,initial)
    ids=[];prev=None
    en_pair=method=='lexicon' and LEXICON[pair][1][0]=='en'
    model_ids=hinglish_tags([words])[0] if romanized_hindi(words,method) else None
    for i,w in enumerate(words):
        start=initial[i] if initial is not None else i==0
        if not any(c.isalpha() for c in w):l=prev  # numbers are language-neutral: inherit the previous language
        elif en_pair and not start and w=='I':l='en'
        elif method=='script':l=script_lid(w,lang)
        elif method=='hindi':
            l=script_lid(w,lang)
            if l=='en' and model_ids is not None:l=model_ids[i]
        elif ' ' in w:  # pyvi compound: merged only when not all-English, so it is Vietnamese ('an toàn')
            l=LEXICON[pair][0][0]
        else:l=lexicon_lid(w,pair,prev)
        ids.append(l)
        if l is not None:prev=l
    return ids

def causal_lid_sequence(words,lang,pair,method,ids,initial=None):
    """Input-feature LIDs: the LID of word t computed from words[:t+1] only, for train and test alike.
    Script LID needs no context, so it returns ids (the label LIDs); Latin-script pairs use the left-context
    rule; for romanized Hindi the model is run on each prefix, all prefixes in one batch."""
    if method=='lexicon':return lid_sequence(words,lang,pair,method,initial=initial,context='left')
    if method!='hindi' or not words:return ids
    # whether the utterance counts as romanized is also decided from the prefix: a Devanagari word later in
    # the sentence must not change how earlier words are tagged
    dev_seen=[];seen=False
    for w in words:
        seen=seen or script_lid(w,'Hindi')=='Hindi';dev_seen.append(seen)
    need=[t for t in range(len(words)) if not dev_seen[t]]
    tags=dict(zip(need,hinglish_tags([words[:t+1] for t in need]))) if need else {}
    out=[];prev=None
    for t,w in enumerate(words):
        if not any(c.isalpha() for c in w):l=prev
        else:
            l=script_lid(w,lang)
            if l=='en' and not dev_seen[t]:l=tags[t][t]
        out.append(l)
        if l is not None:prev=l
    return out
