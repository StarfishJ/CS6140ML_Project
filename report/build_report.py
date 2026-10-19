"""Build report.tex and fig_f1.pdf for the Project Update 1 report. Every number comes from the result files:
  ../github_repo/results/loader_baseline_results.json   baselines (test split)
  ../github_repo/split_manifest.json                    split counts
  report_numbers.json                                    audit: labels, Japanese units, subwords, prefix check
Usage: python build_report.py   then   latexmk -pdf report.tex
"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parent/'github_repo'
B=json.loads((REPO/'results/loader_baseline_results.json').read_text())
M=json.loads((REPO/'split_manifest.json').read_text())
N=json.loads((HERE/'report_numbers.json').read_text())
GITHUB_URL='https://github.com/StarfishJ/CS6140ML_Project'

FILES={'Arabic_eng.csv':'Arabic','Hindi_eng.csv':'Hindi','Korean_eng.csv':'Korean','Russian_eng.csv':'Russian',
       'Chinese_eng.csv':'Chinese','Can_Eng.csv':'Cantonese','Japanese_eng.csv':'Japanese','Spanish_eng.csv':'Spanish',
       'French_eng.csv':'French','German_Eng.csv':'German','Italian_eng.csv':'Italian','Malay_eng.csv':'Malay',
       'Philippines_eng.csv':'Filipino','Vietnamese_eng.csv':'Vietnamese','German_French.csv':'German--French'}
PAIR={fn:(n if '--' in n else n+'--English') for fn,n in FILES.items()}
JSON_PAIR={fn:p.replace('--','–') for fn,p in PAIR.items()}
SEG={'Arabic':'space','Hindi':'space','Korean':'eojeol','Russian':'space','Chinese':'jieba','Cantonese':'PyCantonese',
     'Japanese':'GiNZA','Spanish':'space','French':'space','German':'space','Italian':'space','Malay':'space',
     'Filipino':'space','Vietnamese':'pyvi','German--French':'space'}
LID={'Arabic':'script','Hindi':'script+MuRIL','Korean':'script','Russian':'script','Chinese':'script','Cantonese':'script',
     'Japanese':'script'}
f3=lambda v:f'{v:.3f}'.lstrip('0') if v<1 else f'{v:.3f}'
pct=lambda a,b:f'{100*a/b:.1f}'

def lab(fn,key,splits=('train','val','test')):return sum(N['label'][fn].get(f'{s}|{key}',0) for s in splits)

# ---------- dataset table
drows=[];unks=[];tot=dict(rows=0,kept=0,pos=0)
for fn,name in FILES.items():
    c=M['counts'][JSON_PAIR[fn]];rows=sum(c.values());kept=c.get('T',0)+c.get('V',0)+c.get('E',0)
    pos=lab(fn,'positions');sw=lab(fn,'switches');d=[lab(fn,f'dur_{k}') for k in (0,1,2)];ds=sum(d)
    unk=lab(fn,'unknown_words')/lab(fn,'words')
    drows.append(f"{PAIR[fn]} & {SEG[name]} & {LID.get(name,'wordfreq')} & {rows:,} & {kept:,} & {pos/1000:.0f}K & {pct(sw,pos)} & "
                 f"{round(100*d[0]/ds)}/{round(100*d[1]/ds)}/{round(100*d[2]/ds)} & {100*unk:.2f}\\\\")
    tot['rows']+=rows;tot['kept']+=kept;tot['pos']+=pos;unks.append(100*unk)
can=M['counts']['Cantonese–English'];hin='Hindi_eng.csv'
rom_rows=lab(hin,'rows_romanized');hin_rows=lab(hin,'rows')
cross=sum(N['cross'].values())

# ---------- results table and figure
pr=B['pair_results']
order=sorted(FILES,key=lambda fn:-(pr[JSON_PAIR[fn]]['word_threshold_pair_tuned']['f1']-pr[JSON_PAIR[fn]]['always_switch']['f1']))
rrows=[]
for fn in order:
    r=pr[JSON_PAIR[fn]];a,l,w=r['always_switch']['f1'],r['lid_threshold_pair_tuned']['f1'],r['word_threshold_pair_tuned']['f1']
    rrows.append(f"{PAIR[fn]} & {f3(a)} & {f3(l)} & \\textbf{{{f3(w)}}} & +{f3(w-a)}\\\\")
mean_l,sd_l,mean_w,sd_w=B['pair_tuned_f1_mean'],B['universality_sd_pair_tuned_f1'],B['word_f1_mean'],B['universality_sd_word_f1']
testn=[pr[p]['word_threshold_pair_tuned']['n'] for p in pr]
best=order[0];worst=order[-1]
kr=N['kor']['korean_endings']

def figure():
    import matplotlib;matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':7.5,'axes.edgecolor':'#52514e','axes.linewidth':0.6,
                         'xtick.color':'#52514e','ytick.color':'#0b0b0b','pdf.fonttype':42})
    names=[PAIR[fn].replace('--English','').replace('--','–') for fn in order][::-1]
    vals={k:[pr[JSON_PAIR[fn]][key]['f1'] for fn in order][::-1] for k,key in
          (('Always switch','always_switch'),('Current LID','lid_threshold_pair_tuned'),('Naive (word)','word_threshold_pair_tuned'))}
    colors={'Always switch':'#2a78d6','Current LID':'#eb6834','Naive (word)':'#1baf7a'}  # validated slots 1-3
    marks={'Always switch':'o','Current LID':'s','Naive (word)':'D'}
    dodge={'Always switch':0.22,'Current LID':0.0,'Naive (word)':-0.22}  # equal values stay visible
    fig,ax=plt.subplots(figsize=(3.33,3.1))
    y=range(len(names))
    for i in y:ax.plot([vals['Always switch'][i],vals['Naive (word)'][i]],[i+0.22,i-0.22],color='#d9d8d4',lw=1,zorder=1)
    for k in vals:ax.scatter(vals[k],[i+dodge[k] for i in y],s=14,marker=marks[k],color=colors[k],edgecolor='white',linewidth=0.6,zorder=3,label=k)
    ax.set_yticks(list(y));ax.set_yticklabels(names);ax.set_xlim(0,0.72);ax.set_xlabel('Switch-class F1 (test split)')
    ax.grid(axis='x',color='#e8e7e3',lw=0.5);ax.set_axisbelow(True)
    for s in ('top','right'):ax.spines[s].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(0.4,1.0),ncol=3,frameon=False,handletextpad=0.2,columnspacing=0.8,fontsize=6.8)
    fig.tight_layout(pad=0.3);fig.savefig(HERE/'fig_f1.pdf');plt.close(fig)
figure()

# ---------- Japanese units, subwords, prefix check, punctuation
ja=N['ja']
jarows='\n'.join(f"{name} & {ja[k]['positions']/1000:.0f}K & {100*ja[k]['switch_rate_test']:.1f} & {f3(ja[k]['current_lid'])} & {f3(ja[k]['current_word'])}\\\\"
                 for k,name in (('fugashi','Morphemes (fugashi)'),('sudachi-C','Sudachi mode C'),('ginza','Bunsetsu (GiNZA)')))
def sub(fn,model,key):
    c=N['subword'][f'{fn}|{model}'];g,k=key.split('|');return c.get(f'{g}|{k}',0)/max(1,c.get(f'{g}|words',0))
kor_mbert=sub('Korean_eng.csv','bert-base-multilingual-cased','other|split');en_mbert=sub('Korean_eng.csv','bert-base-multilingual-cased','en|split')
zh_spill=sub('Chinese_eng.csv','xlm-roberta-base','all|spill');ja_spill=sub('Japanese_eng.csv','xlm-roberta-base','all|spill')
pfx=N['prefix']
lid_full=[fn for fn,v in pfx.items() if v['lid_input_identical']==1.0];lid_rest=[fn for fn in pfx if fn not in lid_full]
word_full=[fn for fn,v in pfx.items() if v['word_input_identical']==1.0];word_rest=[fn for fn in FILES if fn not in word_full]
word_min=min(v['word_input_identical'] for v in pfx.values());lid_min=min(v['lid_input_identical'] for v in pfx.values())
names=lambda fns:', '.join(FILES[fn] for fn in fns[:-1])+' and '+FILES[fns[-1]] if len(fns)>1 else FILES[fns[0]]
pc=[(lab(fn,'punct_switches',('test',))/max(1,lab(fn,'punct_positions',('test',))),lab(fn,'punct_switches',('test',))/lab(fn,'switches',('test',))) for fn in FILES]
p_after=(min(a for a,_ in pc),max(a for a,_ in pc));share=(min(b for _,b in pc),max(b for _,b in pc))

tex=r'''\documentclass[sigconf]{acmart}
\setcopyright{none}
\settopmatter{printacmref=false, printfolios=true, authorsperrow=4}
\renewcommand\footnotetextcopyrightpermission[1]{}
\makeatletter\def\@affiliationfont{\small\normalfont}\makeatother
\usepackage{booktabs}
\usepackage{CJKutf8}
\newcommand{\ko}[1]{\mbox{\begin{CJK}{UTF8}{mj}#1\end{CJK}}}
\begin{document}
\title{Predictive Multitask Learning for Streaming Code-Switching}
\subtitle{CS6140 Machine Learning --- Project Update 1 Report}
\author{Huanhuan Yan}\affiliation{\institution{Northeastern University}\city{Seattle}\country{USA}}\email{echoyanxx@hotmail.com}
\author{Jiaming Li}\affiliation{\institution{Northeastern University}\city{Seattle}\country{USA}}\email{ohhh668@gmail.com}
\author{Yuchen Huang}\affiliation{\institution{Northeastern University}\city{Seattle}\country{USA}}\email{huang.yuchen2@northeastern.edu}
\author{Zaixun Li}\affiliation{\institution{Northeastern University}\city{Seattle}\country{USA}}\email{zaixunli0915@gmail.com}
\renewcommand{\shortauthors}{Yan et al.}
\begin{abstract}
We study anticipatory code-switching: after reading the prefix $x_1,\dots,x_t$, predict whether the next word
$x_{t+1}$ switches language and how long the new-language span will last. This report describes the data pipeline
we built for all 15 language pairs of SwitchLingua --- language-specific word segmentation, word-level language
identification, switch and duration labels, a deduplicated 70/10/20 split, and a streaming loader whose inputs
exclude $x_{t+1}$ (up to a small segmentation effect we measure) --- and our first baselines. A naive model that only sees the current word reaches a mean
switch-class F1 of '''+f3(mean_w)+r''' across pairs (SD '''+f3(sd_w)+r'''), roughly twice a model that only knows the
current word's language. Code: \url{'''+GITHUB_URL+r'''}.
\end{abstract}
\maketitle
\pagestyle{plain}

\section{Problem Description}
Code-switching is the alternation between two or more languages within an utterance or conversation. Most NLP work
recognizes a switch after the mixed-language text has been observed. We instead study \textbf{anticipatory}
code-switching: at word $x_t$, using only the prefix $x_1,\dots,x_t$, predict (1) the switch label
$y_{sw}(t)=1$ if $x_{t+1}$ is in a different language than $x_t$, and (2) the duration label $y_{dur}(t)$, the length
of the new-language span starting at $x_{t+1}$: 0 for 1--2 words, 1 for 3--6, 2 for 7 or more, and $-1$ (ignored by
the loss) when there is no switch. Here a token $x_t$ is a word, not a model subword: a whitespace word in
space-separated languages and a segmenter word or bunsetsu in Chinese, Cantonese and Japanese (Section~3); labels
are defined per word. Real-time systems such as bilingual predictive keyboards, dialogue agents and
streaming speech recognition need to know a switch is coming before it happens.
Our central question is \textbf{universality}: does one method anticipate switches equally well across typologically
different pairs? We measure it with the same pipeline and metric on every pair, reporting per-pair switch-class F1
(``Anticipatory F1'') and its standard deviation across pairs. Our final model will be a multilingual transformer
(XLM-R or mBERT) with a causal attention mask and two heads, one for $y_{sw}$ and one for $y_{dur}$.

\section{Background and Related Work}
\textbf{Code-switch prediction.} Solorio and Liu~\cite{solorio2008learning} addressed predicting where a bilingual
speaker will switch between Spanish and English, which no earlier work had automated. They trained Naive Bayes and
VFI classifiers on language-ID, part-of-speech and lexical features of the one or two words before each candidate
point, in a 40-minute conversation of about 8k words. Although the switches it generated were rated nearly as natural
as human ones, Naive Bayes reached only 0.28 F1, on one small corpus and one pair.

Ostapenko et al.~\cite{ostapenko2022speaker} addressed predicting, before it happens, whether the next word in an
English--Spanish dialogue switches language. They fine-tuned XLM-R on the utterance prefix and previous turns of the
Bangor Miami corpus, with prompts describing each speaker's age, origin and language preferences. Although the
prompts raised test accuracy from 72.0\% to 79.7\%, the study covers one pair and a small dataset and does not predict
how long a switch lasts.

Winata et al.~\cite{winata2019synthetic} addressed the lack of English--Mandarin code-switched text for training
language models. They used a pointer-generator network that copies words from machine-translated parallel sentences
and learns when to switch, generating 270K synthetic sentences. Although pre-training on these lowered SEAME test
perplexity from 65.71 to 59.74, perplexity at switch points stayed very high, and only one pair was studied.

\textbf{Language identification in code-switched text.} Solorio et al.~\cite{solorio2014overview} addressed identifying
the language of each word in code-switched social-media text, through the first shared task on Spanish--English,
Nepali--English, Mandarin--English and Modern Standard Arabic--dialectal Arabic. Seven teams submitted systems
ranging from rule-based character n-gram methods to SVMs and CRFs. Although the best F-measure reached 97.7\% for
Nepali--English, it was only 41.7\% on one Arabic test set, so closely related varieties were much harder.

Molina et al.~\cite{molina2016overview} addressed the same problem in a second shared task, on Spanish--English and
Modern Standard--Egyptian Arabic tweets. Nine teams took part, mostly with CRF sequence labelers; two used CNN or LSTM
networks. Although the best token-level F-measure was 97.3\% for Spanish--English, it was only 87.6\% for the Arabic
pair, so closely related languages remained harder.

Aguilar et al.~\cite{aguilar2020lince} addressed evaluating code-switching models on one centralized benchmark across
language pairs and tasks. They built LinCE from ten corpora over four pairs and four tasks (language ID, POS tagging,
NER and sentiment) and reported BiLSTM, ELMo and multilingual BERT baselines. Although mBERT reached up to 98.5\%
language-ID accuracy, it scored only 84.1\% on the Arabic pair, and every task labels complete sentences rather than
predicting ahead.

Khanuja et al.~\cite{khanuja2020gluecos} addressed evaluating NLP models on code-switched English--Hindi and
English--Spanish text. They built GLUECoS with six tasks, from language ID to a new NLI task, and compared
cross-lingual embeddings, mBERT, and mBERT further trained with masked-LM on synthetic code-switched text. Although
the modified mBERT was best on most tasks, NLI reached only 63.1\%, little above chance, on only two pairs.

\textbf{Multilingual encoders.} Devlin et al.~\cite{devlin2019bert} addressed pre-training language representations
that transfer to many tasks with little task-specific design. They pre-trained BERT, a Transformer encoder, with
masked language modeling and next-sentence prediction, then fine-tuned it with one extra output layer per task.
Although it set new results on eleven tasks (GLUE 80.5), every token attends to both left and right context, so
prefix-only prediction needs a causal mask.

Pires et al.~\cite{pires2019multilingual} addressed how well multilingual BERT transfers across languages. They
fine-tuned it for NER and POS tagging in one language, evaluated it zero-shot on others, and tested POS tagging on
code-switched Hindi--English written in Devanagari and in romanized form. Although transfer worked best between
typologically similar languages, accuracy fell from 86.6\% to 50.4\% when the Hindi was romanized, which motivates our
separate romanized-Hindi language ID.

Conneau et al.~\cite{conneau2020xlmr} addressed cross-lingual representation learning for many languages, including
low-resource ones. They trained XLM-R, a masked language model, on 2.5\,TB of filtered CommonCrawl text in 100
languages, including romanized Hindi. Although it beat mBERT by 14.6 points of average XNLI accuracy, adding languages
dilutes per-language capacity, and like BERT it is bidirectional.

Muller et al.~\cite{muller2021unseen} addressed how multilingual models handle languages absent from their
pre-training. They fine-tuned mBERT for POS tagging, parsing and NER on 15 such languages, with and without extra
masked-LM training on raw text, and transliterated hard non-Latin-script languages into Latin script. Although
transliteration raised Uyghur NER from 34.7 to 55.2 F1, it hurt Mingrelian, so the script strongly affects transfer.

\textbf{Multitask learning and data.} Liu et al.~\cite{liu2019mtdnn} addressed improving language understanding with
multitask learning. They proposed MT-DNN, which shares BERT encoder layers across tasks and adds task-specific heads
for classification, similarity regression and ranking, trained jointly. Although it raised the GLUE score to 82.7
(+2.2 over BERT), all its tasks are English and see the full input, unlike our prefix-only switch and duration heads.

Xie et al.~\cite{xie2025switchlingua} addressed the lack of large, multilingual code-switching data. They built
SwitchLingua with LinguaMaster, a GPT-4o multi-agent loop constrained by syntactic switch-point rules and speaker
metadata, producing 420K text samples across 12 languages plus over 80 hours of recorded speech. Although it is the
largest open code-switching resource in linguistic diversity, all of its text is synthetic.

\section{Dataset}
\textbf{Source.} We use the text part of SwitchLingua (SwitchLingua\_text on Hugging Face): 15 pairs,
'''+f'{tot["rows"]:,}'+r''' rows, 14 X--English pairs plus German--French. Each row is a generated conversation whose
\texttt{data\_generation\_result} field is a list of utterances, with metadata such as topic and switch type. There are
no word-level language labels, so we derive them. We use all 15 pairs rather than a subset because universality is
best tested on the widest spread: they cover 7 scripts, 7 language families, SVO, SOV and VSO word order, and one
pair (German--French) without English.

\textbf{Segmentation and word language ID.} Word units and language ID depend on the writing system
(Table~\ref{tab:data}). Space-separated languages are split on whitespace and punctuation, keeping combining marks
(an early version of our regex split 44.5\% of Devanagari words in the first 1,000 Hindi rows into fragments). Chinese, Cantonese and Japanese are
segmented with jieba, PyCantonese and GiNZA bunsetsu units (a content word plus its particles, comparable to the
Korean eojeol); Vietnamese syllables are merged into words by pyvi. A word mixing Latin and another script
(Korean ``presentation-eseo'') is split at the script boundary, and a hyphenated word whose parts are clearly in different
languages (``nag-meeting'') is split at the hyphen. For non-Latin pairs the Unicode script gives the language. Hindi
utterances containing Devanagari use script as well; the remaining '''+pct(rom_rows,hin_rows)+r'''\% of Hindi rows are
fully romanized and are tagged with \texttt{muril-hinglish-lid}, a MuRIL model fine-tuned on LinCE Hindi--English.
For the eight Latin-script pairs, a word gets the language in which it is clearly more frequent according to
wordfreq~\cite{speer2022wordfreq} (Zipf difference $\geq$1.0, or 0.5 toward English for Malay and Filipino, whose
lists contain many English loans), or the language indicated by letters only one side uses. wordfreq combines frequencies from up to seven sources
(Wikipedia, subtitles, news, Google Books, OSCAR web text, Twitter and Reddit); Malay, Filipino and Vietnamese
have only three. Ambiguous words take the
language of the nearest decided words on both sides when these agree, and otherwise the language in which the
word is more frequent. Numbers inherit the previous word's language. On spot-checked
validation samples (about 400--500 words per Latin-script pair), word-level agreement with our own reading was about
98--99\%; these are weak labels, not gold annotations.

\textbf{Labels and split.} From the word language IDs we compute $y_{sw}$ and $y_{dur}$ for every position whose two
words have a known language (unknown words are '''+f'{min(unks):.1f}--{max(unks):.1f}'+r'''\% of words, mostly ambiguous Filipino and Malay words at a switch that neither word list contains, and
Japanese numbers before any known word). Exact duplicate
rows are removed, rows that share any normalized utterance of at least three words are grouped, and groups are split 70/10/20 into train, validation and
test with a fixed seed per pair; '''+str(cross)+r''' utterances appear in more than one split. Cantonese--English
('''+f'{sum(can.values()):,}'+r''' rows, '''+f'{can.get("D",0):,}'+r''' exact duplicates) is downsampled after splitting
to '''+f'{can["T"]+can["V"]+can["E"]:,}'+r''' rows, the median size of the other pairs.

\textbf{Label example.} In the Korean--English utterance ``I want to | \ko{먹어요 맛있어요} | today'' (bars mark
the language spans), the loader gives, at $x_t=$ \emph{to}, $y_{sw}=1$ because the next word is Korean and
$y_{dur}=0$ because the Korean span has 2 words; at \emph{want}, $y_{sw}=0$ and $y_{dur}=-1$; at \ko{맛있어요},
$y_{sw}=1$ and $y_{dur}=0$, since the English span ``today'' has 1 word. Spans are counted up to the end of the utterance; a switch whose new span reaches a word of
unknown language also gets $y_{dur}=-1$ (at most '''+f'{100*max(v["duration_unresolved"]/(v["duration_unresolved"]+v["duration_labeled"]) for v in B["pair_results"].values()):.1f}'+r'''\% of switches in any pair).

\textbf{Streaming loader and causal inputs.} \texttt{StreamingCSDataset} is a PyTorch \texttt{IterableDataset}: a
generator reads one CSV row at a time and yields one example per word $t$ --- the window $x_{t-31},\dots,x_t$, the
language of $x_t$, $y_{sw}$ and $y_{dur}$, and optionally XLM-R/mBERT subword ids scored at the last subword of $x_t$.
The generator yields examples on demand rather than materializing all word-level training examples in memory. The last word of an utterance yields no example and windows never cross utterances.

\textbf{Causal constraint: status.} Labels may read the whole sentence, since they are the answers; inputs must not
include $x_{t+1}$, in training or testing.
\emph{Implemented:} the naive baselines read only $x_t$ and its language (no punctuation after $x_t$); the text passed to
a tokenizer ends at $x_t$; and the input LID is computed from the prefix only (left context for Latin-script pairs, the
Hinglish model run on each prefix for romanized Hindi).
\emph{Verified:} a prefix-only consistency test re-runs segmentation and LID on the prefix alone for 40 validation
utterances per pair. The LID input is identical in '''+str(len(lid_full))+r''' pairs ('''+names(lid_rest)+r''': '''+f'{100*lid_min:.1f}'+r'''\%)
and the word $x_t$ in '''+str(len(word_full))+r''' pairs.
\emph{Not yet implemented:} the transformer's causal attention mask (Update 2), and strictly prefix-only segmentation
for Chinese, '''+names(word_rest)+r'''.
\emph{Known limitation:} those segmenters see the whole utterance; in the test, Chinese words did not change, and in the other three pairs
'''+f'{100*(1-word_min):.1f}'+r'''\% or fewer of the input words $x_t$ differ from what the prefix alone gives. The inputs
are therefore not yet fully leak-free; the leak is small but not zero.

\begin{table*}[t]
\caption{The 15 SwitchLingua pairs after preprocessing. Positions are word positions with a label (all splits); S/M/L
is the share of switches whose new span has 1--2 / 3--6 / 7+ words; Unk.\ is the share of words with no language.}
\label{tab:data}
\small
\begin{tabular}{lllrrrrcr}
\toprule
Pair & Segmentation & Word LID & Rows & Kept rows & Positions & Switch \% & S/M/L \% & Unk.\ \%\\
\midrule
'''+'\n'.join(drows)+r'''
\midrule
Total & & & '''+f'{tot["rows"]:,}'+r''' & '''+f'{tot["kept"]:,}'+r''' & '''+f'{tot["pos"]/1e6:.2f}'+r'''M & & & \\
\bottomrule
\end{tabular}
\end{table*}

\section{Experimentation and Results}
\textbf{Setup.} All scores are switch-class F1 on the test split ('''+f'{min(testn)/1000:.0f}K--{max(testn)/1000:.0f}K'+r'''
word positions per pair). Always stay never predicts a switch (F1 = 0 on every pair). Always switch predicts a switch
at every word; its F1 depends only on the switch rate $p$, $2p/(1+p)$, and serves as the reference line. Current LID
estimates $P(\text{switch}\mid\text{pair},\text{LID of }x_t)$ on train with add-one smoothing. The naive baseline
estimates $P(\text{switch}\mid\text{pair}, x_t)$ for the word itself, shrunk toward its language's rate with five
pseudo-counts. Both use $x_t$ and its language, computed from the prefix (no punctuation after $x_t$), and their thresholds are tuned on each
pair's validation split. Gain is naive F1 minus always-switch F1.

\begin{table}[t]
\caption{Test switch-class F1, sorted by gain of the naive (current-word) baseline over always switch.}
\label{tab:results}
\small
\begin{tabular}{lrrrr}
\toprule
Pair & Always sw. & Cur.\ LID & Naive & Gain\\
\midrule
'''+'\n'.join(rrows)+r'''
\midrule
Mean (SD) & & '''+f'{f3(mean_l)} ({f3(sd_l)})'+r''' & \textbf{'''+f'{f3(mean_w)}'+r'''} ('''+f3(sd_w)+r''') & \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\includegraphics[width=\columnwidth]{fig_f1.pdf}
\caption{Test switch-class F1 per pair (values in Table~\ref{tab:results}). Pairs are ordered by the naive model's gain
over always switch; the grey bar joins always switch and naive.}
\label{fig:f1}
\end{figure}

\textbf{Results.} Table~\ref{tab:results} and Figure~\ref{fig:f1} show that knowing the word itself, not only its
language, roughly doubles F1: from '''+f3(mean_l)+r''' to '''+f3(mean_w)+r''' on average. The current language alone helps
little: it beats always switch by at most '''+f3(max(pr[p]['lid_threshold_pair_tuned']['f1']-pr[p]['always_switch']['f1'] for p in pr))+r'''
and not at all for Hindi, Filipino and Malay, where switching is frequent and goes both ways. The naive model gains
most for '''+PAIR[best]+r''' (+'''+f3(pr[JSON_PAIR[best]]['word_threshold_pair_tuned']['f1']-pr[JSON_PAIR[best]]['always_switch']['f1'])+r'''):
Korean words with the sentence-final endings \emph{-yo} / \emph{-da} are followed by a switch
'''+f'{100*kr["-요"]["switch_rate"]:.0f}\\% / {100*kr["-다"]["switch_rate"]:.0f}'+r'''\% of the time, against
'''+f'{100*kr["other"]["switch_rate"]:.0f}'+r'''\% for other Korean words. It gains least for '''+PAIR[worst]+r'''
(+'''+f3(pr[JSON_PAIR[worst]]['word_threshold_pair_tuned']['f1']-pr[JSON_PAIR[worst]]['always_switch']['f1'])+r''').
Across pairs the naive F1 has SD '''+f3(sd_w)+r''', so a single current word is far from universal; the spread is a
first measurement of the universality gap a context model must close.

\textbf{Segmentation unit.} The word unit decides how many predictions are scored. Re-running the Japanese pair with
three units (Table~\ref{tab:ja}) shows that fine morphemes add many easy no-switch positions: the switch rate falls
and current-LID F1 halves, while the naive model, which sees the word itself, loses much less. We use bunsetsu units,
comparable to Korean eojeol, so that pairs are scored on similar units.

\begin{table}[t]
\caption{Japanese--English with three segmentation units (all splits / test).}
\label{tab:ja}
\small
\begin{tabular}{lrrrr}
\toprule
Unit & Positions & Switch \% & Cur.\ LID & Naive\\
\midrule
'''+jarows+r'''
\bottomrule
\end{tabular}
\end{table}

\textbf{Subwords and punctuation.} With mBERT, '''+f'{100*kor_mbert:.0f}'+r'''\% of Korean words are split into several
subwords (English: '''+f'{100*en_mbert:.0f}'+r'''\%), and with XLM-R a piece crosses into the next word for
'''+f'{100*zh_spill:.1f}'+r'''\% of Chinese and '''+f'{100*ja_spill:.1f}'+r'''\% of Japanese words. We therefore keep labels
at the word level, score each word at its last subword, and will tokenize word by word. Inputs exclude the
punctuation after $x_t$: in SwitchLingua a sentence-final mark is followed by a switch
'''+f'{100*p_after[0]:.0f}--{100*p_after[1]:.0f}'+r'''\% of the time and precedes '''+f'{100*share[0]:.0f}--{100*share[1]:.0f}'+r'''\%
of all switches, so a model that sees it can catch many inter-sentential switches from the mark alone, and scores
with and without punctuation are not comparable.

\section{Plan for Next}
\begin{itemize}
\item \textbf{Causal dual-head model (Update 2).} XLM-R and mBERT with a causal attention mask, switch and duration
heads on $h_t$, loss $\lambda_1 L_{sw}+\lambda_2 L_{dur}$ with \texttt{ignore\_index}$=-1$; check against the
prefix-by-prefix loader, which must give identical predictions.
\item \textbf{Stronger baselines.} Previous plus current word, logistic regression on prefix features, a
duration-majority baseline, and a variant that sees punctuation, reported separately.
\item \textbf{Label quality.} A small hand-checked gold set per pair to measure word-LID and switch-boundary
accuracy, and incremental segmentation of the prefix for Chinese, Cantonese, Japanese and Vietnamese.
\item \textbf{Analysis.} Per-pair F1 bar chart and SD for each backbone, duration accuracy at true switches, and
inter- versus intra-sentential switches using SwitchLingua's switch-type field; a streaming demo at the end.
\end{itemize}

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}
\end{document}
'''
(HERE/'report.tex').write_text(tex,encoding='utf-8')
print('wrote report.tex and fig_f1.pdf')
