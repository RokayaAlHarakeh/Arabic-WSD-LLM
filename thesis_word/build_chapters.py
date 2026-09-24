# -*- coding: utf-8 -*-
"""Chapter content for the Word build. Imported by build_docx.py."""


# ======================================================================
#  CHAPTER 2 — Transformers and Language Models
# ======================================================================
def build_ch2(D):
    D.h1("Chapter 2 — Transformers and Language Models")
    D.p("This chapter establishes the architectural background required by the rest of the report. It is "
        "deliberately selective. Rather than rederiving the Transformer in full, each section is included "
        "because a later design choice or result depends on it: the attention mechanism explains why a "
        "contextual model can perform word sense disambiguation at all; the specific architecture of "
        "Gemma 2 determines the adapter parameter counts of Chapter 4; tokenizer behaviour on Arabic bounds "
        "the sequence budget and foreshadows the difficulty of adapting to legal Arabic; and the causal "
        "language-modelling objective is the foundation of the continued-pretraining experiment.")

    # ---- 2.1 -------------------------------------------------------
    D.h2("2.1  The attention mechanism")
    D.p("The central problem in word sense disambiguation is that a single surface form carries different "
        "meanings in different contexts. A static word embedding of the kind produced by word2vec or GloVe "
        "assigns exactly one vector to a word type, so the Arabic noun \\ar{نفس} receives an identical "
        "representation in every sentence in which it appears — whether it means *soul*, *self*, *same*, or "
        "*breath*. Under such a representation the disambiguation task is not merely hard; it is ill-posed, "
        "because the input to any downstream classifier is constant across the senses it is meant to "
        "separate.")
    D.p("Self-attention resolves this. It produces a representation of each token that is conditioned on "
        "the entire surrounding sequence, so the same surface form receives different vectors in different "
        "contexts. This single property is what makes the task tractable, and it is the reason the models "
        "studied in this report can be applied to WSD at all.")

    D.h3("2.1.1  Scaled dot-product attention")
    D.p("Let X be a sequence of n token representations of dimension d. Attention projects X into three "
        "matrices — queries, keys and values — using learned weight matrices:")
    D.eq("Q = X·W^Q,    K = X·W^K,    V = X·W^V")
    D.p("The output is a weighted average of the value vectors, where the weight assigned to each position "
        "is determined by the compatibility of its key with the current query:")
    D.eq("Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V",
         "replace with a Word equation (Insert → Equation) for the final version")
    D.p("The scaling factor 1/√d_k is not cosmetic. For large d_k the dot products grow in magnitude "
        "proportionally to √d_k, pushing the softmax into regions where its gradient vanishes. Dividing by "
        "√d_k keeps the pre-softmax logits in a range where the function remains trainable.")
    D.p("Interpreted concretely: for each token, the model computes how relevant every other token in the "
        "sequence is to it, and forms its new representation as a relevance-weighted mixture of the "
        "sequence. A target word therefore inherits information from the words that disambiguate it.")
    D.figure("fig_attention.png",
        "Scaled dot-product attention. The target word attends over the sequence, and its output "
        "representation is a relevance-weighted mixture of the value vectors.")

    D.h3("2.1.2  Multi-head attention")
    D.p("A single attention operation averages over the whole sequence, which tends to blur distinct "
        "relationships. Multi-head attention runs h attention operations in parallel with separate "
        "projections, concatenates their outputs, and applies a final linear map. Each head can specialise "
        "— one may track syntactic dependency, another coreference, another local collocation — and the "
        "representation of a token becomes the union of several distinct views of its context.")

    D.h3("2.1.3  Causal masking")
    D.p("The models used in this report are *decoder-only*: they are trained to predict the next token "
        "given all previous ones. This requires that position i may attend only to positions ≤ i; otherwise "
        "the model could read the answer it is being asked to predict. The constraint is imposed by adding "
        "a mask M to the attention logits before the softmax, with M(i,j) = −∞ for j > i and 0 otherwise. "
        "The −∞ entries become zero after the softmax, so future positions contribute nothing.")
    D.p("This mask is the structural difference between the decoder-only models studied here and the "
        "bidirectional encoders discussed in Section 2.5, and it has a direct consequence for the task: in "
        "a decoder, the representation of a target word is informed only by the text preceding it, whereas "
        "an encoder sees both sides.")

    # ---- 2.2 -------------------------------------------------------
    expand_2_1(D)

    D.h2("2.2  Decoder-only architecture and Gemma 2")
    D.p("All experiments in this report use **Gemma 2-2B**, an open-weight decoder-only model released by "
        "Google. Its concrete configuration is given below, since several later results depend on these "
        "exact numbers.")
    D.table([
        ["Property", "Value"],
        ["Decoder layers", "26"],
        ["Hidden size (d_model)", "2,304"],
        ["Attention heads", "8"],
        ["Key/value heads", "4 (grouped-query attention)"],
        ["Head dimension", "256"],
        ["MLP intermediate size", "9,216"],
        ["Vocabulary size", "256,128"],
        ["Embeddings", "Tied (input and output share one matrix)"],
        ["Total parameters", "≈ 2.61 billion"],
    ], widths=[7, 8])
    D.caption("Architectural configuration of Gemma 2-2B.")

    D.p("Three features of this configuration are worth isolating, because each has a consequence later.")
    D.p("**Grouped-query attention.** Standard multi-head attention allocates one key and one value "
        "projection per query head. Gemma 2 instead shares each key/value pair across two query heads, "
        "using 8 query heads but only 4 key/value heads. The motivation is inference efficiency: the "
        "key–value cache, which dominates memory during autoregressive generation, is halved. The "
        "structural consequence is that the K and V projections are half the width of the Q projection "
        "(4 × 256 = 1,024 against 8 × 256 = 2,048). This asymmetry propagates directly into the adapter "
        "parameter counts computed in Section 4.3, where `k_proj` and `v_proj` are measurably cheaper to "
        "adapt than `q_proj`.")
    D.p("**Logit soft-capping.** Gemma 2 bounds its attention and final logits with a scaled hyperbolic "
        "tangent, cap · tanh(z / cap), using cap = 50.0 for attention logits and 30.0 for the output layer. "
        "This stabilises training, but it also means the forward pass depends on an operation sensitive to "
        "numerical precision. In practice this makes Gemma 2 less forgiving than other families when the "
        "attention implementation or the compute dtype is changed.")
    D.p("**Tied embeddings and a large vocabulary.** With a vocabulary of 256,128 and a hidden size of "
        "2,304, the embedding matrix alone contains approximately **590 million parameters — roughly 23% of "
        "the entire model**. Because input and output embeddings are tied, this single matrix serves both "
        "roles. The significance is that any adaptation strategy which unfreezes the embeddings incurs an "
        "enormous increase in trainable parameters; this is precisely the trade-off analysed when the "
        "`modules_to_save` configuration is discussed for continued pretraining.")

    # ---- 2.3 -------------------------------------------------------
    expand_2_2(D)

    D.h2("2.3  Tokenization and Arabic")
    D.p("Language models do not operate on words. They operate on *subword tokens* produced by a learned "
        "segmentation algorithm, and the properties of that algorithm on a given language determine how "
        "much text fits into a fixed context budget. This section is given more space than is conventional "
        "because the interaction between subword tokenization and Arabic morphology is measurable, is "
        "rarely quantified in the Arabic NLP literature, and is load-bearing for both experiments.")

    D.h3("2.3.1  Subword segmentation")
    D.p("Modern tokenizers sit between two failed extremes. A word-level vocabulary cannot represent words "
        "it never saw in training, which is fatal for a morphologically productive language. A "
        "character-level vocabulary generalises perfectly but produces sequences so long that the quadratic "
        "cost of attention becomes prohibitive.")
    D.p("Byte Pair Encoding (BPE) resolves this by learning a vocabulary from data. Beginning from "
        "individual characters, it repeatedly merges the most frequent adjacent pair into a new symbol, for "
        "a fixed number of merges. Frequent words survive as single tokens; rare words decompose into "
        "frequent fragments. SentencePiece extends this by operating directly on raw text, treating "
        "whitespace as an ordinary character, which removes the dependence on a language-specific "
        "pre-tokenizer. Gemma 2 uses a SentencePiece vocabulary of 256,128 entries.")
    D.p("The critical property is that the vocabulary is learned from a *corpus*. If that corpus is "
        "predominantly English, the merge operations that survive will be those useful for English, and "
        "other languages will be represented less efficiently — not because they are intrinsically harder "
        "to segment, but because they were under-represented when the merges were chosen.")

    D.h3("2.3.2  Why Arabic fragments")
    D.p("Arabic is affected by this more than most languages, for reasons rooted in its morphology, treated "
        "in detail in Section 3.1. In brief, a single Arabic orthographic word routinely bundles proclitics, "
        "a templatic stem, and enclitics. The written form \\ar{وبكتابهم} decomposes into the conjunction "
        "\\ar{و}, the preposition \\ar{ب}, the stem \\ar{كتاب}, and the possessive enclitic \\ar{هم} — four "
        "morphemes in one whitespace-delimited token. English orthography separates most of these functions "
        "into distinct words. The expected consequence is that an Arabic word costs more subword tokens "
        "than an English word. This is quantified below.")

    D.h3("2.3.3  Measuring tokenizer fertility")
    D.p("*Fertility* is the mean number of subword tokens produced per input word:")
    D.eq("fertility = (total subword tokens) / (total whitespace-delimited words)")
    D.p("A fertility of 1.0 means every word is a single token; higher values indicate more fragmentation "
        "and therefore a smaller effective context in words.")
    D.p("To quantify this for the model and data used here, the Gemma 2 tokenizer was applied to the Arabic "
        "text of the WSD dataset and, for comparison, to an English corpus of the same measured extent. The "
        "English reference is WikiText-2, a standard benchmark of Wikipedia prose. Words were delimited by "
        "whitespace and no morphological analyser was used, so the reported figures are a conservative "
        "*lower* bound on true morpheme-level fragmentation.")
    D.table([
        ["Corpus", "Words", "Tokens", "Fertility", "Median"],
        ["Arabic — target-word sentences", "14,879", "33,799", "2.272", "2.333"],
        ["Arabic — dictionary glosses", "57,953", "117,652", "2.030", "2.000"],
        ["**Arabic — all**", "**72,832**", "**151,451**", "**2.079**", "2.100"],
        ["English — WikiText-2 prose", "1,019,340", "1,185,237", "**1.163**", "1.151"],
    ], widths=[6.2, 2.4, 2.4, 2.2, 2.0], align_right={1, 2, 3, 4})
    D.caption("Gemma 2 tokenizer fertility on Arabic and English. Higher values indicate more subword "
              "fragmentation per word.")
    D.figure("fig_fertility.png",
        "Tokenizer fertility by corpus. Arabic sentences, Arabic glosses and English prose, "
        "showing the 1.79× gap.")
    D.keypoint("**An Arabic word costs 2.079 subword tokens; an English word costs 1.163.** The Gemma 2 "
               "tokenizer is therefore **1.79 times less efficient on Arabic than on English**, measured on "
               "this data.")
    D.p("Two secondary observations follow from the same measurement. First, the running sentences fragment "
        "more heavily than the dictionary glosses (2.272 against 2.030). This is consistent with the "
        "morphological account above: the example sentences are ordinary usage carrying full clitic "
        "attachment, whereas glosses are written in a terser citation register that reuses a smaller and "
        "more frequent vocabulary. Second, the median per-text fertility closely tracks the aggregate, "
        "indicating the effect is systematic rather than driven by a few pathological items.")

    D.h3("2.3.4  Consequences for this work")
    D.p("Fertility of 2.079 means that a context window of 1,024 tokens holds approximately 490 Arabic "
        "words, where the same budget would hold roughly 880 words of English. It is important to report "
        "honestly what this did and did not constrain.")
    D.p("On Dataset A it did **not** bind. Measuring the tokenized length of all 9,952 formatted training "
        "examples gives a median of 211 tokens and a maximum of 750. **No example was truncated at the "
        "1,024-token limit**, and in fact every example fits within 768. The configured maximum sequence "
        "length was therefore generous rather than restrictive for this dataset.")
    D.p("The measurement remains load-bearing for two other parts of the report. It is the reason the "
        "alternative benchmark, SALMA (Dataset B), was not used: its sequences reach approximately 4,096 "
        "tokens, and at this fertility the memory required for attention over such sequences exceeds the "
        "compute budget available. It also anticipates the continued-pretraining experiment, where legal "
        "Arabic — with its statute references, fixed formulae, and specialised terminology — is expected to "
        "fragment at least as heavily, so that a fixed token budget covers proportionally less legal text.")

    # ---- 2.4 -------------------------------------------------------
    expand_2_3(D)

    D.h2("2.4  Pretraining and the causal language-modelling objective")
    D.p("Both experiments are trained with variants of a single objective, and the continued-pretraining "
        "chapter is built entirely upon it. It is therefore defined precisely here. Given a token sequence "
        "x = (x₁, …, x_T), a causal language model factorises the joint probability autoregressively:")
    D.eq("P(x) = ∏ over t of  P(x_t | x_<t)")
    D.p("Training minimises the negative log-likelihood of the observed sequence, which for a single "
        "sequence is the mean token-level cross-entropy:")
    D.eq("L = −(1/T) · Σ over t of  log P(x_t | x_<t)",
         "replace with a Word equation for the final version")
    D.p("**Teacher forcing.** During training the model is conditioned on the *ground-truth* prefix rather "
        "than on its own previous predictions. This allows all positions to be computed in parallel in a "
        "single forward pass — the causal mask of Section 2.1 guarantees that no position sees its own "
        "target — and is the reason training is far faster than generation. At inference the model must "
        "instead consume its own outputs, which introduces the familiar exposure mismatch.")
    D.p("**Perplexity.** The standard intrinsic metric is perplexity, the exponentiated mean cross-entropy:")
    D.eq("PPL = exp(L)")
    D.p("Perplexity admits a direct reading: it is the effective number of equally likely choices the model "
        "is deciding among at each step. A perplexity of 20 means the model is, on average, as uncertain as "
        "if selecting uniformly from 20 candidates. Lower is better, and a value of 1 would indicate "
        "perfect prediction.")
    D.p("Two properties matter when perplexity is reported as a result. Perplexity is **not comparable "
        "across tokenizers**, because the quantity being averaged — the token — differs; this is a direct "
        "corollary of Section 2.3. And it is **sensitive to domain**: a model evaluated on text resembling "
        "its training distribution will score well irrespective of any genuine capability. The second "
        "property is the reason the continued-pretraining evaluation does not rely on perplexity alone, and "
        "introduces a cloze probe over held-out documents as an independent measure.")

    # ---- 2.5 -------------------------------------------------------
    expand_2_4(D)

    D.h2("2.5  Encoder and decoder architectures for classification")
    D.p("Two architectural families have been applied to word sense disambiguation, and the distinction "
        "positions the contribution of this report.")
    D.p("**Bidirectional encoders** such as BERT remove the causal mask, so every token attends to the "
        "entire sequence in both directions. Trained with masked language modelling, they produce "
        "representations that are then consumed by a task-specific classification head. For WSD this is a "
        "natural fit: the target word can attend to context on both sides, and the sense inventory maps "
        "onto a fixed label set. This was the dominant approach for Arabic WSD, adopted by AraBERT, "
        "ArabGlossBERT, and the original El-Razzaz method.")
    D.p("**Decoder-only models** retain the causal mask and are trained purely to predict the next token. "
        "They perform classification not by attaching a head but by *generating* the answer as text. The "
        "advantages are that no architectural modification is required and that the same model serves any "
        "task expressible as text; the disadvantages are that the target word is contextualised only by "
        "preceding text, and that the output must be parsed from free-form generation rather than read "
        "from a fixed label space. The consequences of that second point — extraction, parse failure, and "
        "the fallback behaviour of the pipeline — are examined in the results chapter.")
    D.p("The paper replicated in this work asks whether generative decoders can compete with the "
        "established encoder approach on Arabic WSD. This report extends that question in a different "
        "direction: not whether a *larger* decoder competes, but whether a substantially *smaller* one "
        "does, and what the benchmark is actually measuring when it does.")

    expand_2_5(D)

    D.h2("2.6  Summary")
    D.p("This chapter has established the machinery on which the remainder of the report depends. "
        "Self-attention produces context-conditioned token representations, which is the property that "
        "makes word sense disambiguation tractable for a neural model. The specific configuration of "
        "Gemma 2-2B — 26 layers, grouped-query attention with asymmetric projection widths, and a "
        "590-million-parameter tied embedding matrix — determines the adapter parameter arithmetic of "
        "Chapter 4 and the configuration decisions for continued pretraining. A measurement of tokenizer "
        "fertility established that Arabic costs 1.79× more subword tokens per word than English under this "
        "tokenizer, which did not constrain Dataset A but explains the exclusion of Dataset B and "
        "anticipates the behaviour of legal Arabic. Finally, the causal language-modelling objective and "
        "perplexity were defined, together with the two limitations — tokenizer dependence and domain "
        "sensitivity — that motivate the additional evaluation designed for the continued-pretraining "
        "experiment.")


# ======================================================================
#  CHAPTER 3 — Arabic NLP and WSD  (outline + ready tables)
# ======================================================================
def build_ch3(D):
    D.h1("Chapter 3 — Arabic NLP and Word Sense Disambiguation")
    D.p("Chapter 2 established why a contextual model can resolve lexical ambiguity in principle. This "
        "chapter establishes why Arabic produces so much of it in practice, defines the word sense "
        "disambiguation task formally, surveys the resources available for it, and sets up the evaluation "
        "machinery whose behaviour on this particular dataset becomes a result later in the report.")

    # ---- 3.1 -------------------------------------------------------
    D.h2("3.1  Sources of lexical ambiguity in Arabic")
    D.p("Arabic is spoken by several hundred million people, yet it is routinely described as low-resource "
        "in natural language processing. The reason is not scarcity of text but the structure of the "
        "language itself, which multiplies the number of distinct meanings a single written form can carry. "
        "Four properties are responsible, and each contributes to the task addressed in this report.")
    D.table([
        ["Property", "Consequence", "Example"],
        ["Templatic (root-and-pattern) morphology", "One consonantal root generates many surface forms",
         "\\ar{ك-ت-ب} → \\ar{كتاب} (book), \\ar{كاتب} (writer), \\ar{مكتوب} (written), \\ar{مكتبة} (library)"],
        ["Clitic attachment", "One orthographic word contains several morphemes",
         "\\ar{وبكتابهم} = \\ar{و} (and) + \\ar{ب} (with) + \\ar{كتاب} (book) + \\ar{هم} (their)"],
        ["Diacritic omission", "Systematic homography in ordinary writing",
         "\\ar{كتب} = kataba (he wrote) / kutiba (was written) / kutub (books)"],
        ["High polysemy", "Many unrelated senses per lemma",
         "\\ar{نفس} = soul / self / same / breath"],
    ], widths=[4.0, 4.6, 6.4])
    D.caption("Four properties of Arabic that create lexical ambiguity, with examples.")

    D.p("**Templatic morphology.** Arabic derives vocabulary by interleaving a consonantal root, typically "
        "three consonants, with a vocalic pattern. The root \\ar{ك-ت-ب}, carrying a general sense of "
        "writing, generates \\ar{كتاب} (book), \\ar{كاتب} (writer), \\ar{مكتوب} (written), and \\ar{مكتبة} "
        "(library), among many others. The system is productive rather than listed, so a lexicon can never "
        "be complete, and a subword tokenizer trained on a different morphological system will segment "
        "these forms inconsistently — a point measured directly in Section 2.3.")
    D.p("**Clitic attachment.** Function words that English writes separately are attached in Arabic "
        "orthography. Conjunctions, prepositions, the definite article, and pronominal objects and "
        "possessives all bind to the stem, so a single whitespace-delimited word may contain four or five "
        "morphemes. This compounds the tokenization cost and means that whitespace is a poor proxy for "
        "lexical units.")
    D.p("**High polysemy.** Even setting orthography aside, Arabic lemmas frequently carry several "
        "unrelated senses. The example used throughout this report, \\ar{نفس}, means *soul* in one context "
        "and *same* in another; these are not shades of a single meaning but distinct entries in a "
        "dictionary. Resolving between them is precisely the task.")

    # ---- 3.2 -------------------------------------------------------
    expand_3_1(D)

    D.h2("3.2  Diacritic omission: the mechanical cause of the task")
    D.p("Of the four properties, diacritic omission deserves separate treatment, because it is the direct "
        "mechanical cause of the ambiguity this project resolves rather than one contributing factor "
        "among several.")
    D.p("The Arabic script encodes consonants and long vowels. Short vowels, gemination, and the absence of "
        "a vowel are written as *diacritics* — small marks above and below the consonantal skeleton. These "
        "marks are almost entirely absent from ordinary written Arabic. Newspapers, official documents, "
        "web text and legal gazettes are written without them; they survive mainly in the Qur'an, in "
        "poetry, in children's books, and in dictionaries.")
    D.p("The consequence is that a single written form corresponds to several distinct words. The "
        "consonantal skeleton \\ar{كتب} may be read as *kataba* (he wrote), *kutiba* (it was written), or "
        "*kutub* (books) — an active verb, a passive verb, and a plural noun. These are not senses of one "
        "word; they are different words that happen to share a spelling once the vowels are removed.")
    D.keypoint("A human reader resolves this constantly and unconsciously, restoring the missing vowels "
               "from context. **That restoration is the task.** When a model is asked to select the correct "
               "sense of an undiacritised Arabic word, it is being asked to perform the same inference a "
               "reader performs, and to make it explicit as a discrete choice.")
    D.p("This framing also explains why the problem is comparatively acute in Arabic. In a language that "
        "writes its vowels, homography exists but is incidental; in Arabic it is systematic, produced by "
        "the writing system itself and therefore present in essentially every sentence.")

    # ---- 3.3 -------------------------------------------------------
    D.h2("3.3  Orthographic normalisation and its cost")
    D.p("Arabic text also varies in ways that are orthographic rather than semantic. The same word may be "
        "written with different but equivalent characters, and a common preprocessing step is to collapse "
        "these variants.")
    D.table([
        ["Variation", "Forms", "Typical normalisation"],
        ["Alef variants", "\\ar{أ}  \\ar{إ}  \\ar{آ}  \\ar{ا}", "collapse all to bare \\ar{ا}"],
        ["Ta marbuta", "\\ar{ة}  vs  \\ar{ه}", "collapse \\ar{ة} to \\ar{ه}"],
        ["Alef maqsura", "\\ar{ى}  vs  \\ar{ي}", "collapse \\ar{ى} to \\ar{ي}"],
        ["Tatweel (kashida)", "\\ar{ـ} used to stretch letters", "remove entirely"],
        ["Diacritics", "\\ar{َ}  \\ar{ِ}  \\ar{ُ}  \\ar{ّ}  \\ar{ْ}", "remove entirely"],
    ], widths=[3.6, 4.4, 6.0])
    D.caption("Common Arabic orthographic normalisation operations.")
    D.p("Normalisation reduces sparsity and improves string matching, but it carries a cost that is "
        "particularly relevant here. The final row of the table removes exactly the marks that disambiguate. "
        "In a corpus where some diacritics survive — dictionary glosses often retain them — stripping "
        "diacritics destroys signal the model could otherwise have used.")
    D.keypoint("**Design decision taken in this project.** Model input was **not** normalised. The raw "
               "dataset text was passed to the tokenizer exactly as published, preserving any diacritics "
               "present and preserving comparability with the study being replicated. Normalisation was "
               "applied **only** in the offline analyses reported later — the lexical-overlap baseline and "
               "the gloss-similarity measurement — where the objective is string comparison rather than "
               "model input. The specific operations used there are documented alongside those results.")

    # ---- 3.4 -------------------------------------------------------
    expand_3_3(D)

    D.h2("3.4  Arabic language models")
    D.p("Two families of pretrained model are available for Arabic, and the choice between them frames this "
        "work.")
    D.p("**Arabic-specific models** are pretrained predominantly or exclusively on Arabic text. AraBERT is "
        "the most widely used encoder, and CAMeLBERT systematically varies the language variety, model "
        "size, and task type to study their interaction. On the generative side, Jais and ALLaM are "
        "large models trained with a strong Arabic focus. The advantage of this family is alignment: their "
        "tokenizers are fitted to Arabic and therefore fragment it far less, and their pretraining data "
        "matches the target language.")
    D.p("**Multilingual models** such as Gemma, LLaMA and Qwen include Arabic within a much larger "
        "multilingual mixture. Their tokenizers are dominated by other languages — the direct cause of the "
        "1.79× fertility penalty measured in Section 2.3 — but they benefit from vastly greater total "
        "pretraining compute and from the broad reasoning ability that scale confers.")
    D.p("This is a genuine tension rather than a settled question, and the study replicated in this report "
        "acknowledges it explicitly: the authors list the exclusion of Arabic-centric models such as Jais "
        "and ALLaM as a stated limitation of their work, citing stability and resource considerations. The "
        "present report inherits that limitation, since it replicates their Gemma configuration, and "
        "records in its future work that comparing an Arabic-centric base model of similar size would "
        "separate the tokenizer effect from model capacity.")

    # ---- 3.5 -------------------------------------------------------
    D.h2("3.5  Word sense disambiguation")
    D.h3("3.5.1  Task definition")
    D.p("Word sense disambiguation is the task of assigning the contextually correct meaning to an "
        "ambiguous word from a predefined inventory. Formally, given a sentence s, a target word w "
        "occurring in s, and a candidate set C = {c₁, …, c_k} drawn from a sense inventory, the system must "
        "select the c ∈ C that matches the use of w in s.")
    D.p("Two elements of this definition matter for the present work. The *sense inventory* is external and "
        "fixed: the system does not decide what the possible meanings are, only which one applies. And "
        "each sense is usually accompanied by a *gloss* — a short dictionary definition in natural language "
        "— which can be supplied to the model as text. Gloss availability is what makes the task expressible "
        "as a text-selection problem, and therefore what makes generative models applicable to it.")

    D.h3("3.5.2  Approaches")
    D.p("Four broad approaches have been applied, in rough historical order.")
    D.bullet("**Knowledge-based methods** compare the context against each gloss without supervised "
             "training. The canonical example is the Lesk algorithm, which selects the sense whose gloss "
             "shares the most words with the surrounding context. These methods require no labelled data "
             "and are used in this report as a non-neural baseline.")
    D.bullet("**Supervised classification** trains a classifier over features or contextual embeddings, "
             "with one output class per sense. This is effective but requires the label space to be fixed "
             "in advance, which is awkward when the inventory is large or open.")
    D.bullet("**Gloss-based pairwise scoring** reformulates the problem as sentence-pair classification: "
             "the context and a candidate gloss are encoded jointly and scored for compatibility. "
             "GlossBERT introduced this framing and ArabGlossBERT applied it to Arabic. It scales to large "
             "inventories because the gloss, not a class index, carries the sense identity.")
    D.bullet("**Generative selection**, the setting of this report, presents the context and all candidate "
             "glosses in a single prompt and asks a decoder-only model to emit the identifier of the correct "
             "sense. No classification head is added and no architectural change is required, but the answer "
             "must be parsed from free-form text.")

    D.h3("3.5.3  The Arabic WSD landscape")
    D.p("Reported results on Arabic WSD span a wide range, partly because the datasets differ substantially "
        "in construction. On the El-Razzaz benchmark used here, the ORCA evaluation suite reports a best "
        "F1 of 76.68% using AraBERT v2, establishing the encoder baseline. GPTAraEval evaluated ChatGPT on "
        "the same data and reached a best F1 of 53.49% in a three-shot setting, a substantial gap that "
        "illustrates the limits of general-purpose prompting on fine-grained disambiguation. More recently, "
        "AraReasoner reported up to 86.27% F1 with a fine-tuned 14B reasoning model.")
    D.p("On the SALMA corpus, the ArabicNLU 2024 shared task established a Target Sense Verification "
        "baseline at 84.2% accuracy, which no participating system surpassed; the best submission reached "
        "77.82% using a 70B instruction-tuned model with structural prompting. EnhancedBERT offers a "
        "complementary ensemble approach.")
    D.p("The pattern across these results is that supervised adaptation on the target dataset outperforms "
        "prompting a much larger general model. That observation motivates the present work, which pushes "
        "the same logic further down the parameter scale.")

    # ---- 3.6 -------------------------------------------------------
    expand_3_5(D)

    D.h2("3.6  Arabic WSD datasets")
    D.p("Six Arabic WSD resources are available, differing in size, annotation scheme and construction "
        "method. The study replicated here surveys them, and its summary is reproduced below because the "
        "choice of dataset materially shapes what a reported accuracy means.")
    D.table([
        ["Dataset", "Size", "Annotation", "Construction", "Source"],
        ["A — El-Razzaz et al. (2021)", "15.5K senses", "Gloss, binary", "Semi-automatic",
         "MSA dictionary"],
        ["B — SALMA (Jarrar et al., 2023)", "34K tokens", "Relatedness scores", "Manual",
         "News and media"],
        ["C — Kaddoura & Nassar (2024a)", "3.7K sentences", "Sense labelling", "Manual + GPT-3.5",
         "Web, multi-domain"],
        ["D — WSDTN (Saidi et al., 2023)", "27.5K sentences", "Gloss-based", "Fully manual",
         "DHDA dictionary"],
        ["E — KSAA-CAD (2024)", "28K pairs", "Gloss, binary", "Semi-automatic", "CAD dictionary"],
        ["F — Al-Hajj & Jarrar (2021)", "167K pairs", "Gloss, true/false", "Semi-automatic",
         "Arabic Ontology"],
    ], widths=[4.4, 2.6, 3.0, 2.8, 3.0])
    D.caption("Major Arabic WSD datasets, after the survey in the replicated study.")
    D.p("**Dataset A**, used throughout this report, was introduced by El-Razzaz et al. to address the "
        "shortage of public gloss-based Arabic resources. It provides 15,549 senses for 5,347 unique words, "
        "extracted from a Modern Standard Arabic dictionary, and frames disambiguation as a binary decision "
        "between a correct and an incorrect gloss for a word in context. The replicated study partitions it "
        "64/16/20 into 9,952 training, 2,487 development and 3,110 test instances; those splits are adopted "
        "here unchanged so that results remain directly comparable.")
    D.keypoint("The binary construction of Dataset A is not a background detail. It means every instance "
               "presents exactly two candidate glosses, which determines the trivial baseline, bounds what "
               "the task can measure, and — as the results chapter argues — explains why model capacity "
               "yields so little advantage on this benchmark.")
    D.p("**Dataset B**, SALMA, was excluded from this work. It is corpus-based rather than "
        "dictionary-based, uses graded relatedness scores from 1% to 100% rather than binary labels, and "
        "contains many items with five or more candidate senses. It is the harder and arguably more "
        "realistic benchmark, but its sequences extend to roughly 4,096 tokens; combined with the Arabic "
        "fertility measured in Section 2.3, the resulting attention memory exceeds the compute budget "
        "available for this project. The exclusion is a compute constraint, not a judgement about the "
        "dataset, and is recorded in the limitations.")

    # ---- 3.7 -------------------------------------------------------
    expand_3_6(D)

    D.h2("3.7  Evaluation metrics")
    D.p("Let y be the gold labels and ŷ the predictions over a label set C. Accuracy is the proportion of "
        "exact matches. For a single class c, precision, recall and F1 are defined from the confusion "
        "counts:")
    D.eq("P_c = TP_c / (TP_c + FP_c)      R_c = TP_c / (TP_c + FN_c)      F1_c = 2·P_c·R_c / (P_c + R_c)")
    D.p("Two aggregation schemes reduce these per-class scores to a single number. *Micro*-averaging pools "
        "the confusion counts across all classes before computing the metric; in single-label multiclass "
        "classification, where every instance has exactly one prediction and one gold label, micro-F1 is "
        "**identical to accuracy**. *Macro*-averaging instead computes the metric per class and takes an "
        "unweighted mean:")
    D.eq("macro-F1 = (1/|C|) · Σ over c of  F1_c")
    D.p("Macro-averaging exists to protect rare classes. Because every class contributes equally regardless "
        "of how many instances it holds, a model that performs well only on frequent classes is penalised. "
        "This is the desirable behaviour when the label distribution is skewed and minority classes matter, "
        "and it is why macro-F1 is reported alongside accuracy throughout the Arabic WSD literature.")
    D.p("The averaging is taken over a class list, and the identity of that list is a detail that becomes "
        "important later. Standard implementations construct it from the union of the labels appearing in "
        "the gold data and the labels appearing in the predictions, so a system can enlarge the denominator "
        "of its own macro average by predicting labels that occur nowhere in the gold data.")
    D.keypoint("Macro-averaging is informative **only when classes have meaningful support**. Its behaviour "
               "when classes contain a single instance each, and the consequences for the dataset used "
               "here, are examined in the results chapter, where the effect is quantified on the actual "
               "predictions.")

    D.h2("3.8  Summary")
    D.p("Arabic generates lexical ambiguity systematically rather than incidentally, principally because "
        "its writing system omits the short vowels that separate otherwise identical forms; restoring them "
        "from context is exactly the disambiguation task. Preprocessing that normalises orthographic "
        "variation can remove the surviving evidence, which is why model input in this project was left "
        "unnormalised and normalisation confined to offline string-comparison analyses. The available "
        "resources differ substantially in construction, and Dataset A's binary gloss format — exactly two "
        "candidates per instance — is the single structural fact that shapes most of the results reported "
        "later. Finally, the standard metrics were defined, together with the precondition on macro-"
        "averaging whose failure on this dataset becomes a contribution of this work.")


# ======================================================================
#  CHAPTER 4 — Model Adaptation
# ======================================================================
def build_ch4(D):
    D.h1("Chapter 4 — Model Adaptation: From Full Fine-Tuning to QLoRA")
    D.p("Both experiments in this report are QLoRA runs, so this chapter carries the most weight. It is "
        "built as an argument rather than a survey: full fine-tuning is shown to be impossible on the "
        "available hardware, the parameter-efficient alternatives are compared so that LoRA appears as a "
        "justified choice rather than a default, LoRA and quantization are developed in the detail required "
        "to explain the configuration decisions of both experiments, and the chapter closes with the "
        "distinction between training objectives on which the conclusion rests.")

    D.h2("4.1  Full fine-tuning and why it does not fit")
    D.p("Conventional fine-tuning updates every parameter of the model. The obstacle is not the model "
        "itself but the state that gradient-based optimisation requires alongside it. Consider Gemma 2-2B, "
        "with approximately 2.61 billion parameters, trained with the Adam optimiser in mixed precision. "
        "Each parameter requires the storage itemised below.")
    D.table([
        ["Component", "Bytes / parameter", "Gemma 2-2B"],
        ["Weights (fp16)", "2", "5.2 GB"],
        ["Gradients (fp16)", "2", "5.2 GB"],
        ["Adam first moment m (fp32)", "4", "10.4 GB"],
        ["Adam second moment v (fp32)", "4", "10.4 GB"],
        ["fp32 master weights", "4", "10.4 GB"],
        ["**Total**", "**16**", "**≈ 41.7 GB**"],
    ], widths=[7.5, 4.0, 3.5], align_right={1, 2})
    D.caption("Memory required for full fine-tuning of Gemma 2-2B with mixed-precision Adam, "
              "excluding activations.")
    D.p("The GPU available for this project is a Google Colab T4 with 16 GB of memory; the NVIDIA L4 used "
        "in the paper being replicated has 24 GB. Neither is close to sufficient, and the shortfall is not "
        "marginal — it is a factor of roughly 2.6 against the larger card.")
    D.keypoint("The decisive observation is *where* the memory goes. The weights themselves account for "
               "only 5.2 of the 41.7 GB. The remaining 36.5 GB is gradients and optimiser state, and that "
               "state scales with the number of **trainable** parameters, not with the size of the model. "
               "Reducing the trainable parameter count therefore attacks the dominant term directly.")

    expand_4_1(D)

    D.h2("4.2  The parameter-efficient fine-tuning landscape")
    D.p("Several families of methods reduce the trainable parameter count. They are summarised below so "
        "that the selection of LoRA can be justified rather than assumed.")
    D.table([
        ["Method", "What is trained", "Inference cost", "Weakness"],
        ["Adapter layers", "Inserted bottleneck MLPs", "Added latency — extra layers in the forward pass",
         "Increases sequential depth"],
        ["Prefix / prompt tuning", "Virtual token embeddings", "Consumes context length",
         "Hard to optimise; limited capacity"],
        ["BitFit", "Bias terms only", "None", "Very limited capacity"],
        ["(IA)³", "Learned rescaling vectors", "Minimal", "Limited expressivity"],
        ["**LoRA**", "Low-rank matrices", "**Zero after merging**", "Rank constrains the update"],
    ], widths=[3.4, 3.8, 4.0, 3.8])
    D.caption("Parameter-efficient fine-tuning methods.")
    D.p("The deciding property is **mergeability**. Because the LoRA update is a linear term added to an "
        "existing weight matrix, it can be folded back into that matrix after training: the deployed model "
        "is architecturally identical to the base model, with no additional layers and no additional "
        "latency. Adapter layers cannot offer this, and prefix methods permanently consume part of the "
        "context window — a meaningful cost given the tokenizer fertility measured in Section 2.3. "
        "Mergeability is also what allows LoRA to compose cleanly with quantization.")

    expand_4_2(D)

    D.h2("4.3  LoRA")
    D.h3("4.3.1  The low intrinsic rank hypothesis")
    D.p("The theoretical basis predates LoRA itself. Aghajanyan et al. showed that fine-tuning objectives "
        "possess a low *intrinsic dimensionality*: a pretrained model can be adapted to a downstream task "
        "by optimising within a subspace far smaller than its full parameter space, and larger pretrained "
        "models exhibit *lower* intrinsic dimensionality, not higher. The interpretation is that "
        "pretraining has already produced the required capabilities, and fine-tuning largely reorients "
        "rather than rebuilds them. LoRA operationalises this observation by constraining the weight update "
        "to be low-rank by construction.")

    D.h3("4.3.2  The decomposition")
    D.p("For a pretrained weight matrix W₀ of shape d × k, the update is factorised as the product of two "
        "thin matrices:")
    D.eq("W = W₀ + ΔW = W₀ + B·A,     B is d × r,   A is r × k,   r ≪ min(d, k)")
    D.p("W₀ is frozen and receives no gradient. Only A and B are trained. Where full fine-tuning of this "
        "matrix would require d × k trainable parameters, LoRA requires r · (d + k).")
    D.figure("fig_lora.png",
        "The LoRA decomposition. The frozen weight W₀ and the low-rank branch B·A run in parallel; "
        "their outputs are summed, with the branch scaled by α/r.")

    D.h3("4.3.3  Three details")
    D.p("**1. Initialisation.** A is initialised from a random normal distribution and B is initialised to "
        "**zero**. Consequently ΔW = B·A = 0 at step zero, and the adapted model is *exactly* the "
        "pretrained model before any update is applied. Training therefore begins from the pretrained "
        "function rather than from a randomly perturbed version of it, which is why LoRA training is stable "
        "and shows no initial loss spike. This detail is easily overlooked but is the reason the method "
        "works reliably.")
    D.p("**2. Scaling.** The update is applied with a fixed scalar:")
    D.eq("h = W₀·x + (α / r) · B·A·x")
    D.p("The purpose of α is to decouple the choice of rank from the choice of learning rate. Increasing r "
        "adds components to the update; without compensation, the magnitude of ΔW would grow with r and the "
        "effective step size would change, so a rank sweep would require re-tuning the learning rate at "
        "every point. Scaling by α/r holds the update magnitude approximately constant.")
    D.keypoint("Only the **ratio** α/r appears in the forward pass. α is therefore not a second independent "
               "hyperparameter: doubling it is mathematically close to doubling the effective learning rate "
               "of the adapter.")
    D.p("The configuration used in this work sets r = 32 and α = 32, giving a ratio of exactly 1.0. This is "
        "a conservative choice; many published recipes use 2.0.")
    D.p("**3. Zero inference latency.** After training, the merged weight W′ = W₀ + (α/r)·B·A can be "
        "computed once and stored. Nothing is added to the forward pass at inference, which distinguishes "
        "LoRA from adapter layers.")

    D.h3("4.3.4  Parameter arithmetic on Gemma 2-2B")
    D.p("The adapters in this work are attached to all seven linear projections in each decoder layer, at "
        "rank 32. Each module contributes r · (d_in + d_out) trainable parameters.")
    D.table([
        ["Module", "r × (d_in + d_out)", "Parameters"],
        ["q_proj", "32 × (2304 + 2048)", "139,264"],
        ["k_proj", "32 × (2304 + 1024)", "106,496"],
        ["v_proj", "32 × (2304 + 1024)", "106,496"],
        ["o_proj", "32 × (2048 + 2304)", "139,264"],
        ["gate_proj", "32 × (2304 + 9216)", "368,640"],
        ["up_proj", "32 × (2304 + 9216)", "368,640"],
        ["down_proj", "32 × (9216 + 2304)", "368,640"],
        ["**Per layer**", "", "**1,597,440**"],
        ["**× 26 layers**", "", "**41,533,440**"],
    ], widths=[4.0, 6.0, 4.0], align_right={2})
    D.caption("LoRA parameter count for Gemma 2-2B at r = 32, all seven target modules.")
    D.p("The adapter therefore contains **41,533,440 trainable parameters**, approximately 1.6% of the "
        "model's 2.61 billion.")
    D.p("Two structural facts follow and are used later. First, `k_proj` and `v_proj` are cheaper than "
        "`q_proj` and `o_proj` — a direct consequence of grouped-query attention halving the key/value "
        "width, as described in Section 2.2. Second, and more consequentially, the three MLP projections "
        "account for 1,105,920 of the 1,597,440 parameters per layer, or **69% of the adapter**. Since the "
        "feed-forward layers are where factual and lexical knowledge is understood to be stored, this "
        "proportion is directly relevant to the configuration chosen for continued pretraining.")

    D.h3("4.3.5  Limitations, and the bridge to continued pretraining")
    D.p("The rank constraint means LoRA can only express updates lying within an r-dimensional subspace of "
        "the full update space. This is ample for *behavioural* adaptation — learning an output format, a "
        "response convention, or a task interface — because such changes are themselves low-dimensional. "
        "It is more restrictive for *injecting knowledge*, which generally requires moving stored "
        "representations and, when new terminology is involved, the embeddings themselves.")
    D.keypoint("This asymmetry is the reason the two experiments use different configurations. The "
               "supervised fine-tuning run adapts a task format and succeeds at low rank. The "
               "continued-pretraining run must move stored knowledge, and therefore requires higher rank, "
               "full coverage of the linear layers, and — because legal Arabic introduces terminology the "
               "tokenizer fragments heavily — trainable embeddings. That configuration change is a "
               "principled consequence of the rank bound, not an unexplained switch.")
    D.p("A recent refinement, DoRA, decomposes the pretrained weight into magnitude and direction and "
        "applies the low-rank update only to the direction, reporting improved performance at equal "
        "parameter budget. It was not used here, but is noted as a natural extension.")

    expand_4_3(D)

    D.h2("4.4  Quantization")
    D.p("LoRA reduces the optimiser state but leaves the frozen base model in memory at full precision. "
        "Quantization addresses that remaining term, and must be understood before QLoRA can be described "
        "as anything other than magic.")
    D.p("Quantization maps a high-precision tensor onto a small set of discrete levels together with a "
        "scale factor. The simplest scheme, *absmax* quantization, maps a tensor to b-bit integers by "
        "normalising against its largest absolute value. The scale must be stored alongside the integers in "
        "order to reconstruct the tensor.")
    D.p("**Why quantization is block-wise.** Applying a single scale to an entire tensor is fragile. If one "
        "weight is far larger in magnitude than the rest, the scale is dominated by that outlier and every "
        "other weight is compressed into a handful of the available levels, destroying resolution across "
        "the tensor. The remedy is to quantize in **blocks** — typically 64 elements — each with its own "
        "scale, so an outlier degrades only its own block.")
    D.p("This is not a hypothetical concern. Dettmers et al. showed that large transformers develop "
        "*emergent outlier features*: specific hidden dimensions whose activations are orders of magnitude "
        "larger than the rest, appearing consistently once models pass a certain scale. Naive uniform "
        "quantization degrades such models badly, and block-wise schemes exist precisely to contain the "
        "damage.")

    expand_4_4(D)

    D.h2("4.5  QLoRA")
    D.p("QLoRA combines a quantized frozen base with trainable LoRA adapters. It contributed three distinct "
        "techniques, described here in full; Section 4.5.4 states which of them this project actually used.")
    D.h3("4.5.1  4-bit NormalFloat (NF4)")
    D.p("Pretrained neural network weights are approximately zero-centred and normally distributed. A "
        "uniform 4-bit grid spaces its 16 levels evenly across the range, which allocates as much "
        "resolution to the sparsely populated tails as to the dense centre — a poor use of a very small "
        "budget. NF4 instead places its 16 levels at the *quantiles* of a standard normal distribution, so "
        "that each bin carries approximately equal probability mass. For data that is genuinely normally "
        "distributed this is information-theoretically optimal. It is applied block-wise, with a scale per "
        "block.")
    D.h3("4.5.2  Double quantization")
    D.p("Block-wise quantization introduces its own overhead. With one fp32 scale per 64-element block, the "
        "scales cost 32/64 = 0.5 bits per parameter — appreciable when the weights themselves cost only 4. "
        "Double quantization quantizes the scales as well, to 8-bit values in blocks of 256, reducing the "
        "overhead to approximately 0.127 bits per parameter. The saving sounds like bookkeeping; on a "
        "multi-billion-parameter model it is hundreds of megabytes.")
    D.h3("4.5.3  Paged optimizers")
    D.p("Memory consumption during training is not constant. A long sequence, or the recomputation "
        "performed by gradient checkpointing, can produce a transient spike that exceeds available memory "
        "and terminates a run that would otherwise have completed. Paged optimizers use NVIDIA unified "
        "memory to page optimiser state to CPU RAM during such spikes and retrieve it afterwards, trading "
        "throughput for the avoidance of an out-of-memory failure.")
    D.h3("4.5.4  The mechanism, the trade-off, and what was used")
    D.keypoint("The base model weights are **stored** in NF4 and **dequantized on the fly** to a "
               "higher-precision format for each matrix multiplication, then discarded. A full-precision "
               "copy of the weights never persists in memory. Gradients flow *through* the frozen quantized "
               "base into the adapters: the base receives no gradient of its own, but backpropagation still "
               "traverses it in order to reach the adapter parameters, which sit inside the network rather "
               "than on top of it.")
    D.p("QLoRA buys memory with compute. Dequantization is real arithmetic performed on every forward pass, "
        "so at equal batch size **QLoRA is slower than fp16 LoRA**. It is chosen not because it is faster "
        "but because it makes training possible at all on the available hardware.")
    D.table([
        ["Configuration", "Peak memory"],
        ["Full fine-tuning, mixed-precision Adam", "≈ 41.7 GB"],
        ["Frozen 4-bit base alone", "≈ 1.5 GB"],
        ["**QLoRA: 4-bit base + r=32 adapters + 8-bit Adam**", "**≈ 2.5 GB**"],
    ], widths=[10.0, 4.5], align_right={1})
    D.caption("Approximate peak training memory for Gemma 2-2B.")
    D.figure("fig_memory.png",
        "Peak training memory: full fine-tuning against QLoRA, with the 16 GB limit of the available "
        "GPU marked.")
    D.p("The reduction is roughly **sixteen-fold**, and it is the reason a free-tier GPU could train this "
        "model at all. The saving comes overwhelmingly from the elimination of optimiser state for 2.61 "
        "billion parameters, exactly as predicted by the analysis in Section 4.1.")
    D.p("For precision, the configuration in this work uses **NF4 quantization** and **double "
        "quantization**, but **not paged optimizers**. Memory pressure was instead managed with an 8-bit "
        "Adam implementation, which quantizes the optimiser *states* — a related but distinct mechanism. "
        "This distinction is documented in the methodology chapter alongside the full configuration.")

    expand_4_5(D)

    D.h2("4.6  Training objectives")
    D.p("This section is short but is the section on which the conclusion of the report depends. Both "
        "experiments minimise a cross-entropy loss over tokens, yet they are not variants of one procedure: "
        "they differ in the data they consume and in what they change about the model.")
    D.table([
        ["", "Continued pretraining (CPT)", "Instruction SFT"],
        ["Data", "Raw domain text", "(instruction, input, output) triples"],
        ["Loss", "Cross-entropy over all tokens", "Cross-entropy, optionally response-only"],
        ["Changes", "Token distribution, vocabulary, register", "Output format, task behaviour"],
        ["Typical LR", "1e-5 – 5e-5", "1e-4 – 2e-4"],
    ], widths=[2.8, 5.8, 5.8])
    D.caption("Continued pretraining and instruction fine-tuning contrasted.")

    D.h3("4.6.1  Loss masking")
    D.p("In instruction fine-tuning the training sequence contains both the prompt and the desired "
        "response. Two conventions exist: computing the loss over the entire sequence, or masking the "
        "prompt so that gradient is received only from the response tokens.")
    D.p("The run in this work used the **full sequence**, following the original Alpaca recipe. The "
        "consequences can be quantified exactly. Measured with the Gemma 2 tokenizer, the fixed instruction "
        "together with its template wrapper occupies 112 tokens, while the mean complete training example "
        "is 216.1 tokens.")
    D.keypoint("Approximately **52% of every training sequence is a constant prompt**, identical across all "
               "9,952 examples, while the discriminative target — the sense identifier — occupies fewer "
               "than four tokens, under 2% of the sequence.")
    D.p("This is defensible: because the constant portion is identical in every example, its contribution "
        "to the gradient is close to a constant offset, and the informative signal remains the final "
        "identifier. It is nonetheless inefficient, and response-only masking is identified as a "
        "straightforward improvement in the future work of this report.")

    D.h3("4.6.2  Sequence packing")
    D.p("Short training examples waste computation when each occupies its own padded sequence. *Packing* "
        "concatenates several examples into one full-length sequence to eliminate that waste. It carries a "
        "known risk: unless position identifiers are reset and attention is masked at example boundaries, "
        "tokens can attend across the boundary into an unrelated example, contaminating the context.")
    D.p("The run in this work did **not** use packing. Each example occupied its own sequence, truncated at "
        "1,024 tokens, and because the per-device batch size was one, no padding was introduced either. "
        "This can be established from the training artifacts rather than asserted.")
    D.table([
        ["Scenario", "Examples / seq.", "Sequences", "Steps / epoch", "Total steps"],
        ["No packing", "1.00", "8,956", "1,120", "**3,360**"],
        ["Packing at 1,024 tokens", "4.74", "1,889", "237", "711"],
        ["**Observed final checkpoint**", "", "", "", "**3,360**"],
    ], widths=[5.0, 2.6, 2.4, 2.4, 2.6], align_right={1, 2, 3, 4})
    D.caption("Step-count arithmetic distinguishing a packed from an unpacked run. "
              "The observed final checkpoint settles the question.")
    D.p("With 8,956 training examples, a batch size of one and gradient accumulation of eight, an unpacked "
        "run performs 1,120 optimiser steps per epoch and 3,360 over three epochs. Had packing been "
        "enabled, the mean example length of 216.1 tokens would have allowed 1024/216.1 = 4.74 examples per "
        "sequence, reducing the total to approximately 711 steps. The final checkpoint written by the run "
        "is `checkpoint-3360`, matching the unpacked figure exactly and exceeding the packed figure by "
        "precisely the packing factor.")

    D.h3("4.6.3  Catastrophic forgetting")
    D.p("Adapting a model to new data risks degrading capabilities acquired during pretraining. "
        "Parameter-efficient methods largely sidestep this: because the base weights are frozen, the "
        "pretrained function is preserved exactly and the adaptation is confined to a small additive term "
        "that can be detached. This is the theoretical justification for the retention check performed "
        "after continued pretraining, in which the model is re-evaluated on the word sense disambiguation "
        "task to confirm that general capability was not lost.")

    D.h3("4.6.4  Domain-adaptive and task-adaptive pretraining")
    D.p("The practice of continuing pretraining on in-domain text before task fine-tuning was systematised "
        "by Gururangan et al., who distinguish *domain-adaptive pretraining* (DAPT) on a broad domain "
        "corpus from *task-adaptive pretraining* (TAPT) on the unlabelled task data itself, and report "
        "gains from both across several domains. The continued-pretraining experiment in this report is a "
        "DAPT setting: a general Lebanese legal corpus, unrelated to the downstream evaluation task.")

    D.h2("4.7  Summary")
    D.p("Full fine-tuning of Gemma 2-2B requires approximately 41.7 GB of training state against the 16 GB "
        "available, with optimiser state rather than weights as the dominant term. LoRA reduces the "
        "trainable parameter count to 41.5 million — 1.6% of the model — by constraining the update to rank "
        "32, and is preferred over competing parameter-efficient methods because it merges back into the "
        "base weights and therefore adds no inference cost. Quantization reduces the remaining frozen model "
        "to approximately 1.5 GB, and QLoRA combines the two for a peak footprint near 2.5 GB, a "
        "sixteen-fold reduction. Finally, the distinction between continued pretraining and instruction "
        "fine-tuning was established, together with two configuration facts about the run performed here — "
        "unmasked loss and the absence of sequence packing — both verified against the training artifacts "
        "rather than assumed.")


# ======================================================================
#  CHAPTER 5 — Related Work
# ======================================================================
def build_ch5(D):
    D.h1("Chapter 5 — Related Work")
    D.p("This chapter situates the present work. It begins with the study being replicated, because that "
        "study defines the dataset, the splits, the model family and the evaluation protocol adopted here. "
        "It then traces the two research lines that study builds on — encoder-based Arabic WSD and "
        "generative evaluation of Arabic — before stating the specific gap this report addresses.")

    # ---- 5.1 -------------------------------------------------------
    D.h2("5.1  The replicated study")
    D.p("Noureldien, Mohamed and Attallah (University of Khartoum), published at the Third Arabic Natural "
        "Language Processing Conference in 2025, benchmark generative large language models for Arabic word "
        "sense disambiguation under both zero-shot and fine-tuned conditions. Their evaluation covers one "
        "proprietary model, GPT-4o, and three open-weight models — LLaMA 3.1-8B, Qwen 2.5-7B and "
        "Gemma 2-9B — across two public datasets.")

    D.h3("5.1.1  Setup")
    D.p("The open models were evaluated in both settings; GPT-4o only zero-shot, since fine-tuning it is "
        "not available. Supervised adaptation used LoRA with the base model loaded in 4-bit precision on an "
        "NVIDIA L4 GPU. For Dataset A the reported configuration is three epochs, batch size 1 with "
        "eight-step gradient accumulation, learning rate 2 × 10⁻⁴, maximum sequence length 1,024, LoRA rank "
        "32 with α = 32 and dropout 0.05, sequence packing enabled, AdamW_8bit, weight decay 0.01, linear "
        "schedule with 50 warmup steps, gradient checkpointing, and seed 3407.")
    D.p("Training examples were formatted as instruction-style JSON objects containing the sentence, the "
        "target word, the candidate senses with their glosses, and the correct identifier — the format "
        "adopted unchanged in this report.")
    D.keypoint("Two deviations between that recipe and the run performed in this work are documented in the "
               "methodology chapter and should be read alongside the results: **sequence packing was "
               "enabled in the original study but not here** (Section 4.6.2), and the evaluation interval "
               "differs. Neither was a deliberate design change; both were consequences of rebuilding the "
               "training pipeline after a framework failure, and both are reported rather than concealed.")

    D.h3("5.1.2  Results")
    D.p("Their headline results are reproduced below. They form the comparison point for this report.")
    D.table([
        ["Model", "A: zero-shot", "A: fine-tuned", "B: zero-shot", "B: fine-tuned"],
        ["Gemma 2-9B", "65.34 / 50.64", "89.39 / 81.72", "72.46 / 56.45", "87.23 / 67.80"],
        ["LLaMA 3.1-8B", "48.59 / 38.28", "90.42 / 83.20", "54.78 / 39.98", "88.51 / 69.41"],
        ["Qwen 2.5-7B", "67.40 / 53.02", "90.77 / 83.98", "55.99 / 47.97", "82.22 / 63.07"],
        ["GPT-4o", "79.16 / 67.92", "—", "79.55 / 64.23", "—"],
    ], widths=[3.4, 3.0, 3.0, 3.0, 3.0], align_right={1, 2, 3, 4})
    D.caption("Accuracy / macro-F1 reported by the replicated study on Dataset A and Dataset B.")
    D.p("Three observations follow. GPT-4o leads clearly in the zero-shot setting on both datasets. "
        "Parameter-efficient fine-tuning lifts every open model above GPT-4o's zero-shot accuracy, which is "
        "the study's central claim. And within the fine-tuned column for Dataset A the three open models "
        "cluster tightly, between 89.39% and 90.77% — a spread of under one and a half points across "
        "models differing by two billion parameters.")

    D.h3("5.1.3  Their error analysis")
    D.p("The authors classify errors into three categories, a taxonomy adopted directly in this report. "
        "*Invalid outputs* are refusals or hallucinated identifiers that do not appear in the candidate "
        "set. *In-set close* errors occur when the two glosses are near-paraphrases. *In-set distant* "
        "errors occur when the glosses are semantically unrelated, and therefore represent a genuine "
        "failure of comprehension.")
    D.p("Their findings on invalid outputs are striking: LLaMA 3.1-8B produced 638 refusals on Dataset A "
        "and 539 on Dataset B in the zero-shot setting, and Qwen produced 1,277 on Dataset B. After "
        "fine-tuning, invalid outputs disappeared. This explains LLaMA's anomalously low zero-shot "
        "accuracy of 48.59%, which is otherwise difficult to reconcile with its fine-tuned performance.")
    D.p("They also report that fine-tuning sharply reduces distant errors — for LLaMA on Dataset A, from "
        "938 to 291 — and that accuracy falls as the candidate set grows, making Dataset A, with mostly one "
        "or two senses per token, the easier of the two benchmarks.")
    D.p("Their illustrative distant error is worth recording, because this report returns to it. For the "
        "verb \\ar{هرب} in the sentence \\ar{هرب من الحفلة بالنوم}, the correct sense is the figurative "
        "\\ar{هرب من مسئولياته: تنصل منها، تملص منها} (to evade one's responsibilities); their fine-tuned "
        "LLaMA 3.1-8B instead selected the literal \\ar{هرب فلان في الأرض أبعد فيها} (to flee across the "
        "land). The result obtained for that same item in this work is reported in the results chapter.")

    D.h3("5.1.4  Stated limitations")
    D.p("The authors identify three limitations. Their evaluation covers only Modern Standard Arabic, so "
        "the findings may not generalise to dialects. Arabic-centric models such as Jais and ALLaM were "
        "excluded for stability and resource reasons, despite the expectation that they might perform "
        "better. And prompting was deliberately minimal — no few-shot examples and no chain-of-thought — "
        "in order to establish a clean zero-shot baseline.")
    D.p("To these, one methodological observation may be added. The authors state that they did not enforce "
        "deterministic decoding or temperature constraints during zero-shot inference. Their zero-shot "
        "figures are therefore single samples from a stochastic process rather than reproducible point "
        "estimates. The present work uses greedy decoding throughout, which makes its own numbers exactly "
        "reproducible from the saved adapter.")

    # ---- 5.2 -------------------------------------------------------
    D.h2("5.2  Encoder-based Arabic word sense disambiguation")
    D.p("The dataset used in this report originates with El-Razzaz, Fakhr and Maghraby, who introduced it "
        "together with a BERT-based gloss disambiguation method. Their contribution was as much a resource "
        "as a model: prior Arabic WSD work was fragmented across private datasets, and a public gloss-based "
        "benchmark made comparison possible.")
    D.p("ArabGlossBERT, by Al-Hajj and Jarrar, fine-tunes BERT on context–gloss pairs, adapting the "
        "GlossBERT framing to Arabic. The same authors later released a substantially larger resource of "
        "approximately 167,000 context–gloss pairs derived from the Arabic Ontology and Birzeit "
        "lexicographic databases, listed as Dataset F in Section 3.6.")
    D.p("ORCA, a broad Arabic language-understanding benchmark spanning sixty datasets and seven task "
        "types, includes this WSD dataset and reports a best F1 of 76.68% with AraBERT v2. That figure is "
        "the strongest published encoder result on the benchmark and is the natural point of comparison "
        "for any generative approach. EnhancedBERT, by Kaddoura and Nassar, offers a complementary "
        "feature-rich ensemble.")

    # ---- 5.3 -------------------------------------------------------
    expand_5_2(D)

    D.h2("5.3  Generative models evaluated on Arabic")
    D.p("GPTAraEval extended evaluation of general-purpose generative models to Arabic across a wide task "
        "suite, including dialectal varieties, and revealed substantial gaps between Modern Standard Arabic "
        "and dialects. On this WSD dataset, ChatGPT achieved a best F1 of 53.49% in a three-shot setting — "
        "well below the encoder baseline, and a concrete illustration that general instruction-following "
        "does not by itself solve fine-grained sense selection.")
    D.p("AraReasoner evaluated reasoning-oriented models, including the DeepSeek family, across fifteen "
        "Arabic tasks under several prompting and fine-tuning strategies, reporting up to 86.27% F1 on the "
        "same dataset with a fine-tuned 14B model. Together with the results in Section 5.1, this "
        "establishes a consistent pattern: task-specific adaptation of a mid-sized open model outperforms "
        "prompting a substantially larger general one.")

    # ---- 5.4 -------------------------------------------------------
    expand_5_3(D)

    D.h2("5.4  Shared tasks and benchmarks")
    D.p("The ArabicNLU 2024 shared task evaluated word sense disambiguation on the SALMA corpus alongside a "
        "location-mention disambiguation track. Its Target Sense Verification baseline, using a "
        "context window of eleven words, reached 84.2% accuracy and was not surpassed by any participating "
        "system; the best submission achieved 77.82% using a 70-billion-parameter instruction-tuned model "
        "with structural prompting.")
    D.p("That outcome is a useful corrective. A carefully constructed task-specific baseline outperformed a "
        "model roughly thirty-five times larger, which is the same lesson this report draws from a "
        "different direction — that on a well-posed selection task, capacity is not the binding constraint.")

    # ---- 5.5 -------------------------------------------------------
    D.h2("5.5  Parameter-efficient adaptation")
    D.p("The adaptation methods themselves are developed in Chapter 4 and are not restated here. What is "
        "relevant to positioning is that every fine-tuned result cited in this chapter — the replicated "
        "study, AraReasoner, and the shared-task submissions — relies on parameter-efficient adaptation "
        "rather than full fine-tuning, and that all of them apply it at seven billion parameters or above. "
        "The behaviour of these methods at the small-model scale, where the adapter is a larger proportion "
        "of a smaller network, is not addressed in the Arabic WSD literature.")

    # ---- 5.6 -------------------------------------------------------
    expand_5_5(D)

    D.h2("5.6  Summary and gap analysis")
    D.p("Prior work has established gloss-based Arabic word sense disambiguation as a task, produced the "
        "public datasets on which it is measured, and demonstrated that both bidirectional encoders and "
        "large generative decoders can perform it to a broadly comparable standard. The most recent "
        "evidence indicates that supervised adaptation of an open model of seven to nine billion parameters "
        "exceeds both the encoder baseline and zero-shot GPT-4o.")
    D.p("Three gaps remain, and this report addresses each of them.")
    D.keypoint("**First, no prior work evaluates sub-3B models on Arabic word sense disambiguation.** The "
               "published generative results begin at seven billion parameters, leaving open whether that "
               "capacity is necessary or merely what was available. The tight clustering of the fine-tuned "
               "Dataset A results — 89.39% to 90.77% across models differing by two billion parameters — "
               "suggests capacity may not be the operative variable.")
    D.keypoint("**Second, no prior work reports trivial baselines for this dataset.** Accuracies between "
               "89% and 91% are reported without stating what is achievable with no model at all. Because "
               "Dataset A is binary by construction, that floor is not 50%, and establishing it changes how "
               "every published figure on this benchmark should be read.")
    D.keypoint("**Third, the secondary metric is reported without examination.** Macro-F1 appears alongside "
               "accuracy throughout this literature, including in every result cited in this chapter. "
               "Section 3.7 noted the precondition that macro-averaging requires meaningful class support; "
               "whether Dataset A satisfies it has not previously been checked.")
    D.p("Addressing these three gaps is the contribution of the chapters that follow.")


# ======================================================================
#  References  (the General Conclusion now lives in build_appendices.py)
# ======================================================================

def build_references(D):
    D.h1("References")
    D.todo("Word does not manage citations automatically here. Two options: (a) use Word's References tab "
           "→ Manage Sources, and insert citations with Insert Citation; or (b) use Zotero or Mendeley with "
           "the Word plug-in. Either way, verify every entry below against the published paper before "
           "submitting — several were reconstructed and are marked.")
    refs = [
        # --- The replicated study ------------------------------------------------
        "Noureldien, Y., Mohamed, A., Attallah, F. (2025). Zero-Shot and Fine-Tuned Evaluation of "
        "Generative LLMs for Arabic Word Sense Disambiguation. In Proceedings of the Third Arabic "
        "Natural Language Processing Conference, pages 298-305. Association for Computational Linguistics.",
        # --- Arabic WSD datasets and methods -------------------------------------
        "El-Razzaz, M., Fakhr, M. W., Maghraby, F. A. (2021). Arabic Gloss WSD Using BERT. Applied "
        "Sciences, 11(6).",
        "Al-Hajj, M., Jarrar, M. (2021). ArabGlossBERT: Fine-tuning BERT on Context-Gloss Pairs for WSD. "
        "In Proceedings of RANLP 2021, pages 35-43. INCOMA Ltd.",
        "Jarrar, M., Malaysha, S., Hammouda, T., Khalilia, M. (2023). SALMA: Arabic Sense-Annotated "
        "Corpus and WSD Benchmarks. In Proceedings of ArabicNLP 2023, pages 359-369. ACL.",
        "Khalilia, M., Malaysha, S., Suwaileh, R., Jarrar, M., Aljabari, A., Elsayed, T., Zitouni, I. "
        "(2024). ArabicNLU 2024: The First Arabic Natural Language Understanding Shared Task. In "
        "Proceedings of the Second Arabic Natural Language Processing Conference. ACL.",
        "Kaddoura, S., Nassar, R. (2024a). A Comprehensive Dataset for Arabic Word Sense Disambiguation. "
        "Data in Brief, 55:110591.",
        "Kaddoura, S., Nassar, R. (2024b). EnhancedBERT: A Feature-Rich Ensemble Model for Arabic Word "
        "Sense Disambiguation with Statistical Analysis and Optimized Data Collection. Journal of King "
        "Saud University - Computer and Information Sciences, 36(1):101911.",
        "Alshammari, W., Almazrua, A., Al Wazrah, A., Almatham, R., Alhoshan, M., Alosaimy, A. (2024). "
        "KSAA-CAD Shared Task: Contemporary Arabic Dictionary for Reverse Dictionary and Word Sense "
        "Disambiguation. In Proceedings of the Second Arabic NLP Conference, pages 677-685. ACL.",
        "[VERIFY] Saidi, R. et al. (2023). WSDTN: a large-scale manually annotated Arabic WSD corpus "
        "based on the Doha Historical Dictionary of Arabic.",
        # --- Arabic benchmarks and evaluations -----------------------------------
        "Elmadany, A., Nagoudi, E. M. B., Abdul-Mageed, M. (2023). ORCA: A Challenging Benchmark for "
        "Arabic Language Understanding. In Findings of ACL 2023, pages 9559-9586. ACL.",
        "Khondaker, M. T. I., Waheed, A., Nagoudi, E. M. B., Abdul-Mageed, M. (2023). GPTAraEval: A "
        "Comprehensive Evaluation of ChatGPT on Arabic NLP. In Proceedings of EMNLP 2023. ACL.",
        "Hasanaath, A., Alansari, A., Ashraf, A., Salmane, C., Luqman, H., Ezzini, S. (2025). "
        "AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP. arXiv:2506.08768.",
        "Alqahtani, S., Aldarmaki, H., Diab, M. (2019). Homograph Disambiguation Through Selective "
        "Diacritic Restoration. In Proceedings of the Fourth Arabic NLP Workshop, pages 49-59. ACL.",
        # --- Arabic language models ----------------------------------------------
        "Antoun, W., Baly, F., Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language "
        "Understanding. In Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing "
        "Tools, pages 9-15. ELRA.",
        "Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., Habash, N. (2021). The Interplay of Variant, "
        "Size, and Task Type in Arabic Pre-trained Language Models. In Proceedings of the Sixth Arabic "
        "NLP Workshop, pages 92-104. ACL. [CAMeLBERT]",
        "Bari, M. S. et al. (2024). ALLaM: Large Language Models for Arabic and English. arXiv:2407.15390.",
        "[VERIFY] Sengupta, N. et al. (2023). Jais and Jais-chat: Arabic-Centric Foundation and "
        "Instruction-Tuned Open Generative Large Language Models.",
        # --- Architecture and pretrained models -----------------------------------
        "Vaswani, A. et al. (2017). Attention Is All You Need. In Advances in Neural Information "
        "Processing Systems (NeurIPS).",
        "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep "
        "Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, "
        "pages 4171-4186. ACL.",
        "Gemma Team, Google DeepMind (2024). Gemma 2: Improving Open Language Models at a Practical "
        "Size. arXiv:2408.00118.",
        "Grattafiori, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.",
        "Bai, J. et al. (2023). Qwen Technical Report. arXiv:2309.16609.",
        "Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., Sanghai, S. (2023). GQA: "
        "Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In Proceedings "
        "of EMNLP 2023. ACL.",
        # --- Tokenization ----------------------------------------------------------
        "Sennrich, R., Haddow, B., Birch, A. (2016). Neural Machine Translation of Rare Words with "
        "Subword Units. In Proceedings of ACL 2016.",
        "Kudo, T., Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword "
        "Tokenizer and Detokenizer for Neural Text Processing. In Proceedings of EMNLP: System "
        "Demonstrations.",
        "Merity, S., Xiong, C., Bradbury, J., Socher, R. (2017). Pointer Sentinel Mixture Models. In "
        "International Conference on Learning Representations (ICLR). [WikiText-2]",
        # --- Parameter-efficient fine-tuning ---------------------------------------
        "Aghajanyan, A., Gupta, S., Zettlemoyer, L. (2021). Intrinsic Dimensionality Explains the "
        "Effectiveness of Language Model Fine-Tuning. In Proceedings of ACL 2021.",
        "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). "
        "LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.",
        "Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of "
        "Quantized LLMs. In Advances in Neural Information Processing Systems (NeurIPS).",
        "Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix "
        "Multiplication for Transformers at Scale. In NeurIPS.",
        "Houlsby, N. et al. (2019). Parameter-Efficient Transfer Learning for NLP. In Proceedings of ICML.",
        "Li, X. L., Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. In "
        "Proceedings of ACL 2021.",
        "Lester, B., Al-Rfou, R., Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt "
        "Tuning. In Proceedings of EMNLP 2021.",
        "Ben Zaken, E., Goldberg, Y., Ravfogel, S. (2022). BitFit: Simple Parameter-efficient "
        "Fine-tuning for Transformer-based Masked Language-models. In Proceedings of ACL 2022.",
        "Liu, H. et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than "
        "In-Context Learning. In NeurIPS. [(IA)3]",
        "Liu, S.-Y. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. In Proceedings of ICML.",
        "Gururangan, S. et al. (2020). Do Not Stop Pretraining: Adapt Language Models to Domains and "
        "Tasks. In Proceedings of ACL 2020.",
        "Taori, R. et al. (2023). Stanford Alpaca: An Instruction-following LLaMA Model. GitHub.",
        # --- Tooling ----------------------------------------------------------------
        "Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. In "
        "Proceedings of EMNLP: System Demonstrations.",
        "Mangrulkar, S. et al. (2022). PEFT: State-of-the-art Parameter-Efficient Fine-Tuning Methods. "
        "GitHub.",
    ]
    for i, r in enumerate(refs, 1):
        p = D.d.add_paragraph()
        p.paragraph_format.left_indent = __import__('docx').shared.Cm(1.0)
        p.paragraph_format.first_line_indent = __import__('docx').shared.Cm(-1.0)
        p.paragraph_format.space_after = __import__('docx').shared.Pt(4)
        run = p.add_run(f"[{i}]  ")
        run.bold = True
        run.font.size = __import__('docx').shared.Pt(10.5)
        r2 = p.add_run(r)
        r2.font.size = __import__('docx').shared.Pt(10.5)

# ======================================================================
#  CHAPTER 6 — Methodology: Supervised Fine-Tuning
# ======================================================================
def build_ch6(D):
    D.h1("Chapter 6 — Methodology: Supervised Fine-Tuning")
    D.p("This chapter documents the supervised fine-tuning experiment end to end: how the source data was "
        "converted into training examples, what the model actually sees, how it was configured and "
        "trained, how predictions were generated and parsed, and how they were scored. Every value "
        "reported here was recovered from the run's own artifacts — the saved adapter configuration, the "
        "serialised training arguments, and the trainer state — rather than from the source scripts, so "
        "the configuration described is the configuration that ran.")
    D.p("The chapter also records two deviations from the recipe published in the replicated study, and "
        "the implementation failures encountered along the way. Both are reported rather than omitted: "
        "the first because it affects comparability, the second because the diagnosis is itself a result.")

    # ---- 6.1 -------------------------------------------------------
    D.h2("6.1  Pipeline overview")
    D.p("The system is four stages, each consuming the output of the previous one. Separating them means "
        "each can be re-run independently — which proved essential on an unreliable compute platform.")
    D.table([
        ["Stage", "Script", "Input", "Output"],
        ["1. Data staging", "create_finetuning_dataset.py", "Three Dataset-A JSON files",
         "9,952 Alpaca records (JSONL)"],
        ["2. Fine-tuning", "finetuning.py", "JSONL + base model", "LoRA adapter (41.5M parameters)"],
        ["3. Inference", "infer_model.py", "Adapter + test set", "Predictions JSON + debug log"],
        ["4. Evaluation", "eval.py", "Predictions + ground truth", "Metrics report JSON"],
    ], widths=[3.0, 4.4, 3.8, 3.8])
    D.caption("The four-stage supervised fine-tuning pipeline.")
    D.figure("fig_pipeline.png",
        "Pipeline overview: from the three source JSON files to the final metrics report.")

    # ---- 6.2 -------------------------------------------------------
    D.h2("6.2  Data staging")

    D.h3("6.2.1  Source format")
    D.p("Dataset A is distributed as three JSON files per split. The separation is deliberate: a single "
        "gloss is shared by many sentences, so storing definitions once avoids duplication.")
    D.bullet("**`*_set.json`** — the question. Each entry holds a `sentence_id`, the sentence text, and a "
             "list of target words; each word carries a `word_id` and a list of candidate `senses`, given "
             "as numeric identifiers only.")
    D.bullet("**`*_truth.json`** — the answer key. The same structure, but each word carries a "
             "`target_sense` field holding the correct identifier.")
    D.bullet("**`*_dictionary.json`** — the gloss lookup, mapping each `sense_id` to its Arabic "
             "`definition`.")
    D.p("The splits used are those defined by the replicated study, adopted unchanged so that results "
        "remain comparable.")
    D.table([
        ["Split", "Sentences", "Target words", "Distinct gold labels", "Candidates per item"],
        ["train80", "9,952", "9,952", "9,925", "exactly 2"],
        ["dev20", "2,487", "2,487", "2,486", "exactly 2"],
        ["**test**", "**3,110**", "**3,110**", "**3,110**", "**exactly 2**"],
    ], widths=[2.8, 2.6, 2.8, 3.6, 3.2], align_right={1, 2, 3})
    D.caption("Dataset A split statistics, measured directly from the distributed files.")
    D.p("Two properties measured here are used later. Each sentence contains exactly one target word, so "
        "sentences and instances are equinumerous. And **every instance offers exactly two candidate "
        "senses** — verified across all three splits with no exceptions — which follows from the binary "
        "gloss construction described in Section 3.6.")
    D.keypoint("In the test split, the 3,110 instances carry 3,110 **distinct** gold labels: every class "
               "has support of exactly one. This measurement is recorded here as a property of the data; "
               "its consequence for the evaluation metrics is developed in the results chapter.")

    D.h3("6.2.2  Flattening to instruction format")
    D.p("The three files are joined into a single flat file of training examples. For each target word, "
        "the staging script looks up the gold answer in the truth file and each candidate identifier in "
        "the dictionary, and emits one record. Where a candidate identifier has no dictionary entry it is "
        "skipped; where no candidate survives, the literal string `none` is substituted, so the field is "
        "never empty.")
    D.p("The output is **JSON Lines** (`.jsonl`) — one complete JSON object per line, rather than a single "
        "JSON array. The format is used because it streams: a file of any size can be read line by line "
        "without holding the whole structure in memory, and lines can be appended by a process that is "
        "later interrupted. The same property is exploited by the resumable inference described in "
        "Section 6.6.")
    D.p("Each record has three fields, following the Alpaca convention:")
    D.bullet("**`instruction`** — the task description. This string is **identical in all 9,952 records**; "
             "it carries no per-example information.")
    D.bullet("**`input`** — the only field that varies. It holds the sentence, the target word, and the "
             "candidate senses with their glosses.")
    D.bullet("**`output`** — the gold sense identifier, stored as a **string** rather than an integer, "
             "because the model is trained to generate it as text.")
    D.p("The staging produces **9,952 records** from the training split, one per target word.")

    D.figure("fig_prompt_flow.png",
             "Data staging: the three source files are joined and flattened into a single "
             "instruction-format record per target word.")

    D.h3("6.2.3  The prompt the model actually sees")
    D.p("At training time each record is rendered into the Alpaca template and terminated with the "
        "end-of-sequence token. The structure is fixed:")
    D.eq("Below is an instruction … ### Instruction: {instruction} ### Input: {input} ### Response: "
         "{output}<eos>")
    D.p("At inference the same string is constructed with the response section left **empty**, and the "
        "model generates the continuation. The instruction block is the following text, identical "
        "throughout:")
    D.keypoint("*\"You are tasked with performing Word Sense Disambiguation (WSD). Your job is to analyze "
               "the given sentence and identify the correct sense for the target word based on the "
               "context. For each sense, you are provided with a Sense ID and its definition. Using the "
               "context of the sentence, choose the most appropriate sense definition and provide the "
               "corresponding Sense ID.\"*")
    D.p("The input block is assembled in a fixed layout — the sentence in single quotes, the target word "
        "in single quotes, then the candidates as bracketed pairs joined by a comma and a space on a "
        "single line:")
    D.eq("Sentence: '{sentence}'  /  Target Word: '{word}'  /  Possible Senses:  "
         "[Sense ID: {id}, Definition: {gloss}], [Sense ID: {id}, Definition: {gloss}]")
    D.p("A worked example makes the structure concrete. The first item of the test set concerns the verb "
        "\\ar{هرب} in the sentence \\ar{هرب من الحفلة بالنوم} (\"he escaped the party by sleeping\"). Two "
        "candidates are offered: \\ar{هرب فلان في الأرض أبعد فيها}, the literal sense of fleeing across "
        "land, and \\ar{هرب من مسئولياته: تنصل منها، تملص منها}, the figurative sense of evading one's "
        "responsibilities. The gold answer is the figurative sense.")
    D.keypoint("Note what the model is being asked to do. **Both definitions are printed in the prompt.** "
               "The model does not retrieve the meaning of \\ar{هرب} from its parameters; it decides which "
               "of two visible strings fits the context. This observation is developed into the central "
               "interpretive claim of the report.")

    D.h3("6.2.4  The prompt-construction invariant")
    D.p("The input block is constructed **twice** in two separate programs — once by the staging script "
        "when building the training file, and once by the inference script when building the evaluation "
        "prompt. These two constructions must produce byte-identical output.")
    D.p("This was verified directly: the two implementations were applied to all 3,110 test items and "
        "their outputs compared character by character. **Zero mismatches were found.**")
    D.p("The verification is not ceremonial. During development the two implementations drifted — the "
        "separator between candidates differed — and the effect was total: **every prediction became "
        "`none`**. A prompt that differs from the training distribution, even in punctuation, is "
        "out-of-distribution for a fine-tuned model; it ceased to emit a bare identifier and produced "
        "explanatory prose instead, which the extractor could not parse.")
    D.keypoint("The lesson is that a fine-tuned model is fitted to **a string format**, not to an abstract "
               "task. Prompt construction is therefore part of the model artifact and must be version-"
               "controlled with it.")

    insert_data_samples(D)

    # ---- 6.3 -------------------------------------------------------
    D.h2("6.3  Tokenization and the sequence budget")
    D.p("Rendered examples are tokenized with truncation at a maximum length of 1,024 tokens. The full "
        "length distribution was measured over all 9,952 training examples using the Gemma 2 tokenizer.")
    D.table([
        ["Statistic", "Tokens"],
        ["Minimum", "169"],
        ["Median", "211"],
        ["Mean", "216.1"],
        ["90th / 95th / 99th percentile", "247 / 262 / 299"],
        ["Maximum", "750"],
        ["**Exceeding the 1,024 limit**", "**0  (0.00%)**"],
    ], widths=[7.5, 4.5], align_right={1})
    D.caption("Tokenized length of the 9,952 formatted training examples.")
    D.p("**No example was truncated.** The longest is 750 tokens and every example fits within 768, so the "
        "configured limit of 1,024 was generous rather than binding for this dataset. The limit is "
        "nonetheless retained because it matches the replicated recipe.")
    D.p("A second measurement concerns where the tokens go. The fixed instruction together with the "
        "template wrapper occupies **112 tokens**, against a mean example length of 216.1.")
    D.keypoint("Approximately **52% of every training sequence is a constant prompt**, identical across "
               "all 9,952 examples, while the answer — the sense identifier — occupies fewer than four "
               "tokens, under 2% of the sequence.")

    D.figure("fig_tokenlen.png",
             "Distribution of tokenized length over all 9,952 formatted training examples. "
             "The configured limit of 1,024 tokens is never reached.")

    D.h3("6.3.1  Loss computation")
    D.p("Labels are produced by a standard causal-language-modelling collator, which sets the labels equal "
        "to the input identifiers. Loss is therefore computed over the **entire sequence**: the model is "
        "trained to predict the instruction and the input block as well as the answer. This is not "
        "completion-only masking.")
    D.p("The choice follows the original Alpaca recipe and is defensible on the grounds given in "
        "Section 4.6.1 — the constant portion contributes a near-constant gradient, so the discriminative "
        "signal remains the final identifier. It is nonetheless inefficient, and response-only masking is "
        "recorded as future work.")

    D.h3("6.3.2  Sequence packing")
    D.p("Sequence packing was **not** used. Each example occupies its own sequence, and because the "
        "per-device batch size is one, no padding is introduced either: every batch is a single example at "
        "its natural length. This is a deviation from the replicated recipe and is discussed in "
        "Section 6.8; the evidence establishing it is the step count given in Section 6.5.")

    # ---- 6.4 -------------------------------------------------------
    D.h2("6.4  Model and adapter configuration")

    D.h3("6.4.1  Base model and quantization")
    D.p("The base model is Gemma 2-2B, loaded from an ungated weight mirror of the official release. It is "
        "loaded in 4-bit precision with the following configuration.")
    D.table([
        ["Setting", "Value", "Purpose"],
        ["load_in_4bit", "True", "Quantize the frozen base"],
        ["bnb_4bit_quant_type", "nf4", "4-bit NormalFloat (Section 4.5.1)"],
        ["bnb_4bit_use_double_quant", "True", "Quantize the quantization constants (Section 4.5.2)"],
        ["bnb_4bit_compute_dtype", "bfloat16", "Dequantization target for each matmul"],
        ["attn_implementation", "sdpa", "Handles Gemma 2 soft-capping correctly; ~2× faster than eager"],
    ], widths=[4.6, 2.6, 6.6])
    D.caption("Quantization configuration of the frozen base model.")
    D.p("The attention implementation is set explicitly and **matched between training and inference**. "
        "This matters for Gemma 2 specifically: as noted in Section 2.2, its logit soft-capping makes the "
        "forward pass sensitive to the attention backend, and a mismatch between the two stages would "
        "evaluate a model different from the one trained.")
    D.p("Before adapters are attached, the quantized model is prepared for k-bit training. This casts "
        "normalisation layers and the output head to 32-bit precision for numerical stability, enables "
        "gradient-checkpointing hooks, and marks the inputs as requiring gradients so that "
        "backpropagation can reach the adapter parameters. The key–value cache is disabled during "
        "training, because it is incompatible with gradient checkpointing: checkpointing recomputes the "
        "forward pass, and a stale cache would corrupt the recomputation.")

    D.h3("6.4.2  LoRA configuration")
    D.p("The adapter configuration below was read from the saved `adapter_config.json`, which is the "
        "durable record of what was trained.")
    D.table([
        ["Field", "Value", "Effect"],
        ["r", "32", "Rank of the update; capacity of the adaptation"],
        ["lora_alpha", "32", "Scaling α/r = 1.0; adapter contributes at face value"],
        ["lora_dropout", "0.05", "Dropout on the LoRA branch input only"],
        ["bias", "none", "Gemma 2 projections carry no bias terms"],
        ["task_type", "CAUSAL_LM", "Wraps the model for next-token prediction"],
        ["target_modules", "q, k, v, o, gate, up, down_proj",
         "All seven linear projections in each of the 26 layers"],
        ["modules_to_save", "null", "Nothing trained in full precision"],
        ["init_lora_weights", "true", "B initialised to zero; ΔW = 0 at step 0"],
        ["peft version", "0.19.1", "Recorded for reproducibility"],
    ], widths=[3.6, 4.2, 6.0])
    D.caption("LoRA adapter configuration, read from the saved adapter.")
    D.p("`modules_to_save` is empty because the task introduces neither new tokens nor a new output head: "
        "sense identifiers are digit strings already present in the tokenizer's vocabulary, so the "
        "existing embeddings already represent everything required. This keeps the trainable parameter "
        "count at **41,533,440**, approximately **1.6%** of the model, as derived in Section 4.3.4.")

    # ---- 6.5 -------------------------------------------------------
    D.h2("6.5  Training procedure")

    D.h3("6.5.1  The internal split")
    D.p("Before training, the 9,952 records are shuffled with a fixed seed and partitioned 90/10 into "
        "**8,956 training** and **996 held-out** examples. The held-out portion is a *monitoring* set: its "
        "role is to indicate during training whether the model continues to improve on data it is not "
        "being fitted to.")
    D.keypoint("This split is **not** the test set. The 3,110 test instances live in a separate file and "
               "were never seen during training in any form. All reported accuracy figures come from that "
               "file.")

    D.h3("6.5.2  Hyperparameters")
    D.p("The values below were deserialised from `training_args.bin` saved with the final checkpoint, and "
        "therefore describe the run as executed rather than the script defaults.")
    D.table([
        ["Parameter", "Value"],
        ["Epochs", "3"],
        ["Per-device batch size", "1"],
        ["Gradient accumulation steps", "8  (effective batch 8)"],
        ["Learning rate", "2 × 10⁻⁴"],
        ["Scheduler", "linear"],
        ["Warmup steps", "50"],
        ["Optimizer", "adamw_8bit"],
        ["Weight decay", "0.01"],
        ["Max gradient norm", "1.0"],
        ["Precision", "bf16"],
        ["Gradient checkpointing", "enabled, non-reentrant"],
        ["Seed", "3407"],
        ["Logging interval", "every 20 steps"],
        ["Checkpoint interval", "every 100 steps, keeping the last 2"],
        ["Evaluation interval", "every 500 steps"],
    ], widths=[6.6, 5.4])
    D.caption("Training hyperparameters, recovered from the serialised training arguments.")

    D.h3("6.5.3  Step arithmetic")
    D.p("The step count follows directly from the data size and batching, and provides an independent "
        "check on the packing question:")
    D.eq("8,956 examples ÷ (batch 1 × accumulation 8) = 1,120 steps per epoch × 3 epochs = 3,360 steps")
    D.p("The final checkpoint written by the run is named `checkpoint-3360`, matching exactly. Had packing "
        "been enabled, the mean example length of 216.1 tokens would have allowed 1024 ÷ 216.1 = 4.74 "
        "examples per sequence, reducing the total to roughly **711** steps. The observed count is 4.74 "
        "times larger — precisely the packing factor — which confirms that packing was inactive.")

    D.h3("6.5.4  Training and held-out loss")
    D.p("Loss was logged every 20 steps, producing 168 measurements, and the held-out set was evaluated "
        "seven times.")
    D.table([
        ["", "Start", "Epoch 1", "Epoch 2", "Epoch 3 / final"],
        ["Training loss", "2.09", "0.773 (mean)", "0.478 (mean)", "0.279"],
        ["Held-out loss", "0.834", "0.755", "0.658", "0.615"],
    ], widths=[3.4, 2.6, 2.8, 2.8, 3.2], align_right={1, 2, 3, 4})
    D.caption("Training and held-out loss over the three epochs.")
    D.p("Two observations follow. First, the initial training loss of 2.09 is far **below** the value "
        "expected from an untrained forward pass: with a vocabulary of 256,128, a uniformly random "
        "predictor would score ln(256,128) ≈ 12.45. A starting loss of 2.09 therefore confirms the "
        "pretrained model was intact and the adapter was correctly attached — a check that proved "
        "decisive, as Section 6.9 explains.")
    D.p("Second, the held-out loss decreased monotonically across all seven measurements, from 0.834 to "
        "0.615, and never rose. There is thus no evidence that training continued past the point of "
        "benefit. The gap between the final training loss (0.279) and the final held-out loss (0.615) "
        "indicates the model fits the training data more closely than unseen data, which is expected after "
        "three epochs; what matters for the overfitting question is the direction of the held-out curve, "
        "and it improves throughout.")
    D.figure("fig_loss.png",
        "Training and held-out loss across 3,360 optimisation steps.")

    D.h3("6.5.5  Resilience")
    D.p("The compute platform disconnects frequently, so training was made restartable. Checkpoints are "
        "written every 100 steps to persistent storage, bounding the loss from an interruption to roughly "
        "25 minutes of compute. On restart the script scans for checkpoints, **discards any that lack a "
        "complete trainer state** — the signature of a crash during a save — and resumes from the most "
        "recent valid one.")

    # ---- 6.6 -------------------------------------------------------
    D.h2("6.6  Inference")
    D.p("At inference the base model is loaded in 16-bit precision and the trained adapter attached on top. "
        "Quantization is unnecessary here because a 2-billion-parameter model fits the available memory in "
        "half precision without it; a flag is provided to quantize the base for larger models.")

    D.h3("6.6.1  Generation")
    D.p("Generation is configured for determinism.")
    D.table([
        ["Setting", "Value", "Reason"],
        ["do_sample", "False", "Greedy decoding — the highest-probability token at every step"],
        ["max_new_tokens", "128", "Generous; a valid answer is one or two tokens"],
        ["eos_token_id / pad_token_id", "EOS", "Generation halts at the end-of-sequence token"],
    ], widths=[4.6, 2.4, 6.8])
    D.caption("Generation configuration.")
    D.keypoint("Greedy decoding is a deliberate choice for **reproducibility**. Because no sampling is "
               "involved, re-running inference against the same adapter reproduces byte-identical output "
               "and therefore the identical accuracy. Under sampling the headline figure would vary "
               "between runs and could not be quoted as a single number.")

    D.h3("6.6.2  Answer extraction")
    D.p("The model returns free-form text, from which an identifier must be recovered. Two regular "
        "expressions are applied in order: a run of digits followed by the end-of-sequence token, and a "
        "run of digits alone on a line. These cover the output shapes a correctly trained model produces.")
    D.p("A recovered identifier is then **validated against the candidate set** for that item. If it is "
        "absent — or if neither pattern matched — the prediction is recorded as the string `none`.")
    D.keypoint("Because each item offers exactly two candidates, a `none` prediction means the model "
               "either produced unparseable text or invented an identifier that was never offered. Both "
               "are **format failures**, categorically different from selecting the wrong sense. Recording "
               "them separately allows the two failure modes to be counted independently in the results.")

    D.h3("6.6.3  Resumable inference and logging")
    D.p("Inference over 3,110 sentences takes long enough that a disconnection is likely, so results are "
        "persisted incrementally. After every completed sentence the prediction is appended to a JSON "
        "Lines checkpoint and flushed to disk. On restart the checkpoint is read back and completed "
        "sentences are skipped, so an interruption costs at most one sentence. The final predictions file "
        "is assembled at the end in the original test-set order.")
    D.p("In parallel, a debug log records for every target word the complete prompt, the **raw generated "
        "text** before any parsing, and the extracted prediction. This log is the only record of what the "
        "model actually emitted, and it is what makes the format-failure analysis possible.")

    # ---- 6.7 -------------------------------------------------------
    D.h2("6.7  Evaluation")
    D.p("Scoring pairs each prediction with its gold label and computes accuracy together with macro-"
        "averaged precision, recall and F1. Identifiers are compared as strings, and `none` participates "
        "as an ordinary label rather than being excluded — a model that declines to answer is treated as "
        "having answered incorrectly, which is the appropriate convention here.")
    D.keypoint("**A fragility worth recording.** Predictions and ground truth are aligned by **position** "
               "in their respective lists, not by matching sentence and word identifiers. The pipeline is "
               "therefore correct only because inference writes predictions in the order it read the test "
               "set. Reordering the test file would silently compare every prediction against the wrong "
               "gold answer, with no error raised. Aligning on identifiers would be more robust and is "
               "noted as an improvement.")

    # ---- 6.8 -------------------------------------------------------
    D.h2("6.8  Deviations from the replicated recipe")
    D.p("The configuration above follows the published recipe closely but not exactly. The differences are "
        "listed here so that the comparison in the results chapter can be read with them in mind.")
    D.table([
        ["Aspect", "Replicated study", "This work", "Consequence"],
        ["Model", "Gemma 2-9B", "Gemma 2-2B", "Deliberate — the research question"],
        ["Hardware", "NVIDIA L4 (24 GB)", "Free-tier T4 (16 GB)", "Deliberate — the compute constraint"],
        ["Sequence packing", "Enabled", "**Not used**",
         "Incidental; arose when the training stack was rebuilt"],
        ["Evaluation interval", "Every 100 steps", "Every 500 steps", "Incidental; reduces monitoring "
         "resolution, not model quality"],
        ["Decoding", "Not specified for fine-tuned runs", "Greedy, deterministic",
         "Makes results exactly reproducible"],
    ], widths=[2.8, 3.6, 3.2, 4.4])
    D.caption("Differences between the published recipe and the run performed here.")
    D.p("The packing difference deserves comment. Packing concatenates several short examples into one "
        "full-length sequence for efficiency; its absence means the run performed 3,360 optimiser steps "
        "where a packed run would have performed roughly 711. The model therefore received substantially "
        "more gradient updates over the same data. Whether this helped, hurt, or was neutral cannot be "
        "determined without an ablation, which the compute budget did not permit; it is recorded as a "
        "difference rather than claimed as an improvement.")

    # ---- 6.9 -------------------------------------------------------
    D.h2("6.9  Implementation challenges")
    D.p("The first training run produced a model that generated nothing usable. Diagnosing it consumed "
        "most of the project's engineering effort, and the method of diagnosis is worth recording.")

    D.h3("6.9.1  A corrupt adapter, and how it was found")
    D.p("Every prediction from the first fine-tuned model was empty or malformed. Because a generative "
        "pipeline has many places to fail, the components were isolated and tested one at a time.")
    D.table([
        ["Test", "Observation", "Conclusion"],
        ["Training vs inference prompt", "Formatting mismatch", "A real bug, fixed — but not the cause"],
        ["Tokenizer round-trip", "Correct", "Tokenizer is sound"],
        ["Base model under the framework", "Blank output, then a crash", "The framework's forward pass "
         "is broken"],
        ["Base model under plain library", "Fluent text", "The underlying library is sound"],
        ["The trained adapter", "Garbage output", "**The adapter itself is corrupt**"],
    ], widths=[4.4, 3.8, 5.8])
    D.caption("Layer-by-layer isolation of the generation failure.")
    D.p("The root cause was a bleeding-edge dependency stack installed by an unpinned upgrade, whose "
        "implementation of the Gemma 2 forward pass was broken. Training had run *through* that broken "
        "forward pass, so the resulting adapter had been optimised against meaningless outputs.")
    D.keypoint("The diagnostic signature was visible in the logs from the first minutes: the initial "
               "training loss was approximately **25**, against the ≈12.45 expected from a uniformly "
               "random predictor over a 256,128-token vocabulary. **A loss above the random baseline is "
               "not a warm-up artefact — it indicates the forward pass is wrong.** Recognising this "
               "earlier would have saved the run.")

    D.h3("6.9.2  Rebuilding on a stable stack")
    D.p("The acceleration framework was removed entirely — it is a training speed-up and was never "
        "required for correctness — and the pipeline rebuilt on standard components: a 4-bit base model "
        "with a PEFT adapter, trained by the core trainer with a causal-language-modelling collator. The "
        "high-level supervised-fine-tuning wrapper was also dropped after its interface changed, reducing "
        "the dependency surface. All hyperparameters were retained. **Sequence packing was a feature of "
        "the discarded wrapper, and its removal is the origin of the deviation recorded in Section 6.8.**")

    D.h3("6.9.3  Other issues")
    D.bullet("**Training throughput.** Initial training ran at roughly 17 seconds per step, implying about "
             "a day for three epochs. Switching the attention implementation to `sdpa` and making the "
             "evaluation pass configurable brought this within budget.")
    D.bullet("**Session limits.** The free compute tier permits only a few hours per session. Frequent "
             "checkpointing with automatic resume converted disconnections from run-ending events into "
             "minor setbacks.")
    D.bullet("**Storage exhaustion.** Persistent storage filled mid-run, threatening the final save. The "
             "corrupt model from the first attempt was deleted and the storage trash emptied, since "
             "deleted files continue to count against the quota until then.")
    D.bullet("**A dependency conflict.** The adapter library raised a hard error against an unrelated "
             "preinstalled package that the project never uses. A small compatibility guard causes that "
             "code path to be skipped.")
    D.p("Two general lessons follow, and are returned to in the conclusion: unpinned dependencies on a "
        "fast-moving platform are a genuine correctness risk, not merely a reproducibility inconvenience; "
        "and every stage of a pipeline on unreliable infrastructure should be independently restartable.")

    # ---- 6.10 ------------------------------------------------------
    D.h2("6.10  Summary")
    D.p("Three Dataset-A JSON files were joined and flattened into 9,952 instruction-format records, "
        "rendered into a fixed Alpaca template whose construction was verified byte-identical between "
        "training and inference across all 3,110 test items. Examples were tokenized with a 1,024-token "
        "limit that no example reached, without packing and without loss masking. Gemma 2-2B was loaded "
        "in 4-bit NF4 with double quantization and adapted with a rank-32 LoRA over all seven linear "
        "projections — 41,533,440 trainable parameters, 1.6% of the model — for three epochs and 3,360 "
        "optimiser steps, with held-out loss falling monotonically from 0.834 to 0.615. Inference used "
        "greedy decoding with regular-expression extraction validated against the candidate set, and "
        "scoring used accuracy and macro-averaged precision, recall and F1. Two incidental deviations from "
        "the published recipe were recorded, along with the framework failure that caused the first "
        "training run to be discarded.")

# ======================================================================
#  CHAPTER 7 — Results: Supervised Fine-Tuning
# ======================================================================
def build_ch7(D):
    D.h1("Chapter 7 — Results: Supervised Fine-Tuning")
    D.p("This chapter reports the outcome of the supervised fine-tuning experiment. It opens with the "
        "headline metric and its comparison against published results, then does three things the "
        "published literature on this dataset does not: it establishes what can be achieved without a "
        "model at all, it examines whether the secondary metric carries information, and it decomposes the "
        "errors that remain. The chapter closes with an interpretation of why a model of this size "
        "performs as it does.")
    D.p("All figures derive from a single inference pass over the 3,110-instance test split using greedy "
        "decoding, and are therefore exactly reproducible from the saved adapter.")

    # ---- 7.1 -------------------------------------------------------
    D.h2("7.1  Headline result")
    D.table([
        ["Metric", "Value"],
        ["Accuracy", "**90.42%**"],
        ["Macro-precision", "0.8310"],
        ["Macro-recall", "0.8379"],
        ["Macro-F1", "**0.8333**"],
        ["Instances evaluated", "3,110"],
        ["Correct", "2,812"],
        ["Errors", "298"],
    ], widths=[7.0, 5.0], align_right={1})
    D.caption("Final performance of the fine-tuned Gemma 2-2B on the Dataset A test split.")
    D.p("The fine-tuned model answered 2,812 of 3,110 items correctly. The remainder of this chapter is "
        "concerned with what that number means — against what floor it should be read, whether the "
        "macro-averaged figures add anything to it, and what the 298 errors consist of.")

    # ---- 7.2 -------------------------------------------------------
    D.h2("7.2  Comparison with published results")
    D.p("The replicated study reports fine-tuned results for three open models on the same dataset and the "
        "same splits. Placing this work alongside them gives the following picture.")
    D.table([
        ["Model", "Parameters", "Accuracy", "Macro-F1"],
        ["Gemma 2-9B (published)", "9B", "89.39", "81.72"],
        ["LLaMA 3.1-8B (published)", "8B", "90.42", "83.20"],
        ["Qwen 2.5-7B (published)", "7B", "90.77", "83.98"],
        ["**Gemma 2-2B (this work)**", "**2B**", "**90.42**", "**83.33**"],
    ], widths=[5.4, 2.4, 2.6, 2.6], align_right={1, 2, 3})
    D.caption("This work against the fine-tuned results published for Dataset A.")
    D.p("The 2-billion-parameter model matches the 8-billion-parameter result to two decimal places and "
        "exceeds the same-family 9-billion-parameter model by 1.03 accuracy points, using roughly a "
        "quarter of the parameters and a free-tier GPU rather than an NVIDIA L4.")

    D.h3("7.2.1  How much of that difference is real")
    D.p("A single accuracy figure on a finite test set is an estimate, and the comparison above should not "
        "be read without its uncertainty. Treating each item as an independent Bernoulli trial, the "
        "standard error at p = 0.9042 on n = 3,110 is:")
    D.eq("SE = √( 0.9042 × 0.0958 / 3110 ) = 0.0053  =  0.53 percentage points")
    D.p("A 95% confidence interval is therefore approximately **[89.4, 91.4]**.")
    D.keypoint("Every fine-tuned result in the table above falls inside that interval. **Differences of "
               "roughly one point on this test set are not statistically distinguishable**, and the "
               "apparent ranking among the four models should not be treated as meaningful. Establishing "
               "a real difference would require a paired test — McNemar's test on per-item agreement — "
               "which is not possible here because the published per-item predictions are not released.")
    D.p("The defensible claim is therefore not that the smaller model is *better*. It is that it is **not "
        "worse**, which given a fourfold difference in parameters is the more interesting statement, and "
        "one this chapter goes on to explain.")

    # ---- 7.3 -------------------------------------------------------
    D.h2("7.3  Establishing the baselines")
    D.p("An accuracy of 90.42% is only interpretable against a floor. The published literature on this "
        "dataset reports no baselines, so they are established here.")

    D.h3("7.3.1  Trivial baselines")
    D.p("Section 6.2.1 established that every instance in Dataset A offers exactly two candidates. Random "
        "selection therefore scores 50%. But the two candidates are not interchangeable: measuring the "
        "position of the gold answer across all three splits gives a strong and consistent asymmetry.")
    D.table([
        ["Baseline", "train80", "dev20", "test"],
        ["Random choice between the two candidates", "50.00", "50.00", "50.00"],
        ["Always select the first candidate", "34.26", "35.06", "35.05"],
        ["**Always select the second candidate**", "**65.74**", "**64.94**", "**64.95**"],
        ["Simplified Lesk (ties → first)", "47.86", "48.49", "47.81"],
        ["Simplified Lesk (ties → second)", "—", "—", "64.98"],
        ["**Fine-tuned Gemma 2-2B (this work)**", "—", "—", "**90.42**"],
    ], widths=[7.0, 2.4, 2.4, 2.4], align_right={1, 2, 3})
    D.caption("Baselines on Dataset A, computed directly from the distributed files.")
    D.keypoint("A one-line program that always selects the second candidate — which cannot read Arabic and "
               "has no model — achieves **64.95%**. The consistency of this figure across the three splits "
               "(65.74 / 64.94 / 64.95) shows it is a property of how the dataset was **constructed**, not "
               "a sampling artefact of the test set.")
    D.p("The consequence for how this work should be reported is direct. The gain attributable to the "
        "model is **+25.47 points over 64.95%**, not +40 points over 50%. The same correction applies to "
        "every published accuracy on this benchmark.")
    D.figure("fig_baselines.png",
        "Accuracy against the trivial baselines. The bar chart makes the true floor visible.")

    D.h3("7.3.2  A non-neural baseline: Simplified Lesk")
    D.p("The trivial baselines exploit dataset structure but use no linguistic information at all. The "
        "classical non-neural method for this task is the Lesk algorithm, which selects the gloss sharing "
        "the most words with the context. If a simple lexical-overlap method performed well, the value of "
        "a neural approach would be far less clear.")
    D.p("Tokens were normalised before comparison: diacritics and tatweel removed, alef, ya and "
        "ta-marbuta variants unified, punctuation stripped, and single-character tokens discarded. No "
        "morphological analyser was used, so these figures are a conservative estimate.")
    D.table([
        ["Measurement", "Value"],
        ["Items where the two glosses tie (no discriminating overlap)", "2,010  (64.6%)"],
        ["…of which both glosses share nothing with the sentence", "1,299  (41.8%)"],
        ["Items where Lesk has an actual signal", "1,100  (35.4%)"],
        ["Accuracy on those 1,100 items", "68.09%"],
        ["**Lesk overall, ties resolved to the second candidate**", "**64.98%**"],
        ["For comparison: always select the second candidate", "64.95%"],
    ], widths=[8.4, 3.6], align_right={1})
    D.caption("Simplified Lesk on the test split, decomposed.")
    D.keypoint("Lexical overlap contributes **0.03 percentage points** over the positional rule. On this "
               "dataset the classical method is, to three significant figures, worthless.")
    D.p("The explanation lies in the data. Arabic dictionary glosses are short — a mean of 9 tokens — and "
        "written in a citation register, whereas the example sentences are ordinary usage. The two rarely "
        "share surface forms, so in 64.6% of items there is no overlap difference to decide with, and in "
        "41.8% neither gloss shares anything with the sentence at all.")
    D.p("This is a useful negative result rather than a disappointing one. It establishes that the "
        "benchmark cannot be solved by string matching, which means the neural result is not a disguised "
        "lexical overlap. Section 7.5.5 provides direct evidence for the same conclusion.")

    # ---- 7.4 -------------------------------------------------------
    D.h2("7.4  The macro-F1 metric is degenerate on this dataset")
    D.p("Macro-F1 is reported alongside accuracy throughout the Arabic WSD literature, including in every "
        "result cited in Chapter 5. Section 3.7 noted the precondition: macro-averaging is informative "
        "only when classes have meaningful support. This section shows that Dataset A does not satisfy it, "
        "and that the reported macro-F1 therefore carries no information beyond accuracy.")

    D.h3("7.4.1  The structural cause")
    D.p("Measuring the test split directly: it contains **3,110 instances and 3,110 distinct gold "
        "labels**. Every gold class has support of exactly one. This follows from the construction of the "
        "dataset — each item is a distinct dictionary sense — and is not a peculiarity of the split.")
    D.p("Standard implementations compute the macro average over the **union** of the labels appearing in "
        "the gold data and those appearing in the predictions. Any label a system predicts that is not "
        "the gold answer of some item is therefore admitted to the class list as a class with no true "
        "instances, scoring zero on every metric.")
    D.keypoint("A system can enlarge the denominator of its own macro average simply by being wrong in a "
               "**varied** way. Each distinct incorrect identifier it emits creates a new zero-scoring "
               "class.")

    D.h3("7.4.2  The arithmetic, verified")
    D.p("Because every real class holds exactly one item, its recall is 1 if that item was answered "
        "correctly and 0 otherwise. Macro-recall therefore reduces to the number of correct answers "
        "divided by the size of the class list:")
    D.eq("macro-recall = (number correct) / |labels in y_true ∪ y_pred|")
    D.p("Substituting the observed values:")
    D.listing([
        'correct           = 0.9042 x 3110   =  2,812',
        'reported recall   =                    0.8379',
        '',
        '2,812 / 0.8379    =  3,356.0        <-- exact, to one decimal place',
        '3,356 - 3,110     =    246          <-- classes present only in predictions',
    ], caption="Recovering the macro-average denominator from the reported metrics.")
    D.p("The denominator resolves to 3,356.0 exactly. Subtracting the 3,110 genuine gold classes leaves "
        "**246 phantom classes** — sense identifiers that appear only because the model predicted them. "
        "This is consistent with the error structure: the 298 errors produced 246 distinct incorrect "
        "identifiers, some repeating.")
    D.p("Equivalently, macro-recall is accuracy multiplied by a shrinkage factor:")
    D.eq("macro-recall = accuracy × (3,110 / 3,356) = 0.9042 × 0.9267 = 0.8379")

    D.h3("7.4.3  Why this makes the metric uninformative")
    D.p("Consider two hypothetical systems, each making exactly 298 errors and therefore each achieving "
        "identical accuracy.")
    D.table([
        ["System", "Errors", "Distinct wrong labels", "Denominator", "Macro-recall"],
        ["A — always wrong the same way", "298", "1", "3,111", "0.9039"],
        ["B — wrong in 298 different ways", "298", "298", "3,408", "0.8251"],
    ], widths=[5.0, 1.8, 3.0, 2.4, 2.6], align_right={1, 2, 3, 4})
    D.caption("Two systems with identical accuracy and materially different macro-recall.")
    D.keypoint("Identical accuracy, nearly eight points of macro-recall between them — determined entirely "
               "by whether the errors were varied. **Macro-F1 on this dataset is accuracy rescaled by a "
               "quantity the model controls but which reflects nothing about its quality.**")
    D.p("The same structure is visible in the published results. Every Dataset A macro-F1 in Table 5.1 "
        "sits roughly seven points below its matching accuracy, which is the signature of this effect "
        "rather than a property of the models.")
    D.p("Macro-F1 is nonetheless reported in this work, for comparability with the literature. The "
        "recommendation is that it should be accompanied by the caveat established here, and that "
        "accuracy should be treated as the only informative metric on this benchmark.")

    # ---- 7.5 -------------------------------------------------------
    D.h2("7.5  Error analysis")
    D.p("The 298 errors were joined back to the source files and examined along the axes used by the "
        "replicated study, together with several the study does not consider.")

    D.h3("7.5.1  Error taxonomy: no format failures")
    D.p("The replicated study classifies errors as **invalid outputs** (refusals or identifiers not in the "
        "candidate set), **in-set close** errors between near-paraphrases, and **in-set distant** errors "
        "between unrelated glosses. Applying that taxonomy here:")
    D.table([
        ["Outcome", "Count", "Share"],
        ["Correct", "2,812", "90.42%"],
        ["**Invalid output** (`none`, or an identifier not offered)", "**0**", "**0.00%**"],
        ["In-set error (the other valid candidate)", "298", "9.58%"],
    ], widths=[7.4, 2.2, 2.4], align_right={1, 2})
    D.caption("Error taxonomy applied to the fine-tuned model's 3,110 predictions.")
    D.keypoint("The model produced **zero malformed or out-of-set outputs across all 3,110 items**. It "
               "never refused, never hallucinated an identifier, and never emitted text the extractor "
               "could not parse. **Every error is genuine sense confusion.**")
    D.p("This is a substantive result rather than a technicality. The replicated study reports that "
        "LLaMA 3.1-8B produced 638 refusals on this dataset in the zero-shot setting — 20.5% of the test "
        "split — which is what depresses its zero-shot accuracy to 48.59%, below the trivial baseline. "
        "Their invalid outputs also disappeared after fine-tuning, so the finding here corroborates "
        "theirs: instruction fine-tuning solves the output-format problem completely, and does so at 2B "
        "as reliably as at 8B.")
    D.p("It also means the extraction machinery described in Section 6.6.2 — the two regular expressions "
        "and the candidate-set validation — was never actually needed for this model. That is worth "
        "stating: the fragility identified during development did not materialise in the final run.")

    D.h3("7.5.2  Positional bias")
    D.p("Section 7.3.1 established that the gold answer is the second candidate 64.95% of the time. A "
        "model could exploit this without understanding anything, so the question is whether it did.")
    D.table([
        ["", "Dataset", "Model"],
        ["Proportion answering \"second candidate\"", "64.95%", "**61.35%**"],
    ], widths=[7.0, 2.6, 2.6], align_right={1, 2})
    D.caption("The model's positional preference against the dataset's prior.")
    D.p("The model selects the second candidate **less** often than the data warrants — 3.6 points below "
        "the prior. A system that had absorbed the positional regularity would sit at or above it. "
        "Splitting accuracy by where the gold answer lies sharpens the point:")
    D.table([
        ["Gold answer position", "Items", "Errors", "Error rate"],
        ["First candidate (the minority case)", "1,090", "93", "**8.53%**"],
        ["Second candidate (the majority case)", "2,020", "205", "10.15%"],
    ], widths=[6.0, 2.2, 2.2, 2.6], align_right={1, 2, 3})
    D.caption("Accuracy decomposed by the position of the gold answer.")
    D.keypoint("The model is **more** accurate when the answer sits in the rarer first position. A "
               "positionally biased model would show the opposite. There is no evidence that the reported "
               "accuracy is inflated by exploitation of the dataset's construction.")

    D.figure("fig_position.png",
             "Positional preference of the model against the dataset's prior. The model "
             "selects the second candidate less often than the data warrants.", width_cm=12.5)

    D.h3("7.5.3  Close versus distant errors")
    D.p("Following the replicated study's taxonomy, errors were binned by the lexical similarity of the "
        "two candidate glosses, measured as Jaccard overlap on normalised tokens. High similarity "
        "indicates near-paraphrases, where confusion is understandable; zero similarity indicates "
        "semantically unrelated options, where it is not.")
    D.table([
        ["Gloss-pair similarity", "Items", "Errors", "Error rate"],
        ["0.00 — no shared tokens", "1,167", "119", "10.20%"],
        ["0.00 – 0.10", "1,202", "110", "9.15%"],
        ["0.10 – 0.25", "624", "55", "8.81%"],
        ["0.25 – 0.50", "85", "9", "10.59%"],
        ["0.50+ — near-duplicates", "32", "5", "**15.62%**"],
    ], widths=[5.4, 2.2, 2.2, 2.6], align_right={1, 2, 3})
    D.caption("Error rate by lexical similarity of the two candidate glosses.")
    D.p("The near-duplicate bin is the worst, as expected — but it holds only 32 items. Across the other "
        "3,078 the error rate is essentially flat, between 8.8% and 10.6%. The mean gloss-pair similarity "
        "of errors (0.0686) is barely different from that of correct answers (0.0667).")
    D.keypoint("**Gloss similarity does not explain the errors.** Near-paraphrase pairs are harder, but "
               "they are too rare to account for the error mass; the remaining errors are spread evenly "
               "across all similarity levels. This is reported as a negative result: the hypothesis that "
               "errors concentrate in genuinely hard near-synonym cases is not supported.")

    D.h3("7.5.4  What does predict errors")
    D.p("Three other factors show clearer effects.")
    D.p("**Whether the gloss restates the target word.** Many Arabic glosses open by repeating the lemma "
        "in a usage frame — \\ar{ضاع ماله} for the verb \\ar{ضاع} — while others are purely abstract, such "
        "as \\ar{تواد ومحبة} for \\ar{أخوة}. The difference is large:")
    D.table([
        ["", "Items", "Error rate"],
        ["Target word appears within the gold gloss", "2,220", "**7.57%**"],
        ["Target word does not appear", "890", "**14.61%**"],
    ], widths=[6.6, 2.4, 2.6], align_right={1, 2})
    D.caption("Error rate by whether the correct gloss restates the target word.")
    D.p("The error rate nearly doubles when the anchor is absent. This is intuitive — a repeated lemma "
        "gives the model a direct lexical bridge between prompt and gloss — but it also suggests part of "
        "the task is easier than it appears, since two-thirds of items carry that bridge.")
    D.p("**Word frequency.** Items whose target word occurs only once in the test set are harder than "
        "those whose word recurs (11.05% against 6.38% error), consistent with familiarity acquired during "
        "training.")
    D.p("**Sentence length**, unexpectedly, is inversely related to accuracy: error rates rise from 8.22% "
        "on the shortest items to 16.67% on the longest. More context normally helps. The likely "
        "explanation is that in this dictionary-derived corpus the short items are simple usage examples "
        "while the long ones are literary or Quranic quotations carrying archaic vocabulary, so length is "
        "acting as a proxy for difficulty rather than for information. This is offered as a hypothesis; "
        "the 12-plus-word bin contains only 24 items.")
    D.figure("fig_errors.png",
        "Error rate by gloss-pair similarity, lemma anchoring, word frequency and sentence length.")

    D.h3("7.5.5  Direct evidence against lexical matching")
    D.p("Section 7.3.2 showed that lexical overlap performs no better than a positional rule *on average*. "
        "A stronger test is available: identify the items where overlap would actively mislead, and see "
        "how the model behaves on them.")
    D.p("There are **351 items** in which the *incorrect* gloss shares more tokens with the sentence than "
        "the correct one does. These are traps: any surface-matching method chooses wrongly by "
        "construction.")
    D.keypoint("**The model answered 309 of those 351 correctly — 88.0%.** It resists the surface cue "
               "precisely when the surface cue is wrong.")
    D.p("Taken with the Lesk result, this is the strongest available evidence that the model is performing "
        "semantic selection rather than sophisticated string matching. The overall pattern is consistent: "
        "error rates are 6.81% where lexical overlap points to the right answer, 10.20% where it is "
        "uninformative, and 11.97% where it points to the wrong one — a real effect, but far too small to "
        "suggest the model is relying on it.")

    D.h3("7.5.6  Qualitative examples")
    D.p("Three errors illustrate the categories above, and one non-error is worth recording.")
    D.p("**A frequency error.** For the sentence \\ar{ضاع البخور} (\"the incense \\ar{ضاع}\"), the gold sense is "
        "\\ar{ضاعت الرائحة طابت، فاحت، انتشرت} — the scent spread. The model selected \\ar{ضاع ماله تبدد، "
        "زال} — his money was lost. The verb \\ar{ضاع} has a common sense (to be lost) and a rare one used "
        "specifically of fragrance; the subject \\ar{البخور}, incense, should force the rare reading. The "
        "model took the frequent sense and missed the selectional constraint.")
    D.p("**An unsolvable item.** For the target \\ar{سلي}, the two candidate glosses are \\ar{سلي / سلي} "
        "and \\ar{سلي}. Neither carries semantic content and the two are not distinguishable by any means. "
        "The model's answer here is a coin flip.")
    D.p("**A near-duplicate.** For \\ar{رزم}, the candidates are \\ar{رزم الورق ونحوه رزمه؛ جمعه في شيء "
        "واحد وشده} (gathered and tied) and \\ar{رزم الورق ونحوه جمعه في شيء واحد ولفه} (gathered and "
        "wrapped). These are genuine near-synonyms; a fluent reader would hesitate.")
    D.keypoint("**The item the replicated study uses as its own failure case.** For \\ar{هرب} in "
               "\\ar{هرب من الحفلة بالنوم}, the study reports that its fine-tuned LLaMA 3.1-8B chose the "
               "literal \\ar{هرب فلان في الأرض أبعد فيها} over the correct figurative \\ar{هرب من "
               "مسئولياته}, and presents this as an illustrative distant error. **The model trained in "
               "this work answered that item correctly.**")
    D.p("A single item proves nothing on its own, and it is reported as an illustration rather than as "
        "evidence. It does, however, sit consistently with the aggregate picture: a model with a quarter "
        "of the parameters is not systematically weaker on the cases the larger model finds hard.")

    D.h3("7.5.7  The achievable ceiling")
    D.p("Some items in this dataset cannot be answered correctly by any system. Measuring gloss-pair "
        "similarity across the test split identifies five items whose two candidate glosses are identical "
        "after normalisation, differing only by a redundant repetition of the lemma — the \\ar{سلي} case "
        "above is one of them.")
    D.p("These are annotation artefacts. A perfect model could not exceed approximately 99.8% on this test "
        "split, and any system that answers these five correctly has done so by chance. The effect is "
        "small, but it should be noted before an unexplained residual error rate is attributed to the "
        "model.")

    # ---- 7.6 -------------------------------------------------------
    D.h2("7.6  Interpretation: why parameter count matters so little here")
    D.p("The results above pose a question. A 2-billion-parameter model matches an 8-billion one and "
        "exceeds a 9-billion one from the same family; the three published fine-tuned results span 1.38 "
        "accuracy points across a fourfold range of parameters. Scale is doing almost nothing. Why?")
    D.p("The answer follows from the structure of the benchmark, established in Section 6.2 and visible in "
        "the worked example there. **Every item supplies both candidate glosses in the prompt.** The model "
        "is never required to recall what an Arabic word means; the meanings are given to it as text. Its "
        "task is to decide which of two visible strings is compatible with the surrounding sentence.")
    D.keypoint("This is **discrimination between two supplied options**, not **knowledge retrieval**. "
               "Parameter count buys stored knowledge. A benchmark that supplies the knowledge in the "
               "prompt does not reward the thing that scale provides.")
    D.p("Three findings in this chapter support that reading. The task is not solvable by lexical overlap "
        "(Section 7.3.2), so it does require semantic comparison. The model resists misleading surface "
        "cues in 88% of trap cases (Section 7.5.5), so the comparison is genuine. And it exhibits no "
        "positional shortcut (Section 7.5.2), so the accuracy is not an artefact of dataset construction. "
        "What remains is a comparison of two short strings against a context — a low-dimensional operation, "
        "which is also why a rank-32 adapter over 1.6% of the parameters was sufficient to learn it.")
    D.p("This interpretation carries directly into the second half of the project. Supervised fine-tuning "
        "here taught a **task format**: read two options, choose one, emit its identifier, stop. It added "
        "no knowledge, because none was needed. Continued pretraining on a specialised corpus addresses "
        "the opposite problem, and the contrast between them is developed in the discussion.")

    # ---- 7.7 -------------------------------------------------------
    D.h2("7.7  Threats to validity")
    D.p("Five caveats apply to the results in this chapter and are stated here rather than left implicit.")
    D.bullet("**Single run, single seed.** All figures come from one training run with seed 3407. No "
             "variance estimate over restarts is available, so the reported accuracy carries an unmeasured "
             "run-to-run component in addition to the sampling error quantified in Section 7.2.1.")
    D.bullet("**No paired significance test.** The comparison against published models rests on "
             "overlapping confidence intervals only. A McNemar test would be conclusive but requires "
             "per-item predictions that have not been released.")
    D.bullet("**A configuration deviation.** Sequence packing was enabled in the published recipe and not "
             "here (Section 6.8), so this run performed roughly 4.7 times more optimiser steps over the "
             "same data. Whether that helped is untested.")
    D.bullet("**Baselines are computed, not published.** The trivial and Lesk baselines were computed for "
             "this work from the distributed files. The Lesk implementation uses whitespace tokenisation "
             "without morphological analysis, so it is a conservative lower bound on what a tuned "
             "knowledge-based system might achieve.")
    D.bullet("**One dataset, one language variety.** All conclusions concern Dataset A, a dictionary-"
             "derived Modern Standard Arabic benchmark with exactly two candidates per item. The "
             "discrimination-versus-retrieval interpretation in Section 7.6 depends on that binary "
             "structure and should not be extended to benchmarks with larger candidate sets without "
             "retesting.")

    # ---- 7.8 -------------------------------------------------------
    D.h2("7.8  Summary")
    D.p("The fine-tuned Gemma 2-2B reaches 90.42% accuracy on the 3,110-instance test split, matching the "
        "published 8-billion-parameter result and exceeding the same-family 9-billion-parameter model, "
        "though all four results fall within a 95% confidence interval of [89.4, 91.4] and should be "
        "regarded as indistinguishable.")
    D.p("Three findings go beyond replication. The trivial baseline for this dataset is **64.95%**, not "
        "50%, and a classical lexical-overlap method adds 0.03 points to it — so the model's contribution "
        "is +25.47 points, and the benchmark is not solvable by string matching. The macro-F1 figure "
        "reported throughout the literature is **degenerate** on this dataset: with every gold class "
        "holding a single instance, it reduces to accuracy scaled by a denominator the model inflates "
        "through the variety of its errors, verified exactly as 2,812 ÷ 3,356. And the error analysis "
        "shows **zero format failures**, no positional bias, and 88% accuracy on 351 items constructed to "
        "mislead a surface matcher.")
    D.p("Together these support the interpretation that this benchmark measures discrimination between two "
        "supplied definitions rather than retrieval of stored knowledge, which explains why four models "
        "spanning 2 to 9 billion parameters perform within 1.4 points of one another.")


"""Data-transformation listings inserted into Chapter 6 §6.2.

Every sample below is the ACTUAL first item of the Dataset-A test split
(sentence_id 32768), traced through each stage of the pipeline.
"""


def insert_data_samples(D):
    D.h3("6.2.5  Worked example: one item through every stage")
    D.p("The transformations described above are easier to follow on a concrete case. This section traces "
        "a single item — the first entry of the test split — from the three source files through to the "
        "score it contributes. The item concerns the Arabic verb \\ar{هرب}, and it is the same item the "
        "replicated study uses to illustrate a distant error.")

    # ---- Stage 0: the three source files ----------------------------
    D.p("**Stage 0 — the raw sources.** Three separate files must be joined. The set file supplies the "
        "question and the candidate identifiers, but no glosses; the truth file supplies the answer; the "
        "dictionary supplies the text of each candidate.")
    D.listing([
        '// test_set.json  — the question (glosses are NOT here)',
        '{',
        '  "sentence_id": 32768,',
        '  "sentence": ":-\\ar{هرب من الحفلة بالنوم.}",',
        '  "words": [ { "word_id": 5089,',
        '               "word": "\\ar{هرب}",',
        '               "senses": [14704, 14706] } ]',
        '}',
    ], caption="Source file 1 — the question. Candidate senses appear as bare identifiers.",
        arabic_lines={3, 5})

    D.listing([
        '// test_truth.json  — the answer key (same shape, plus target_sense)',
        '{',
        '  "sentence_id": 32768,',
        '  "words": [ { "word_id": 5089, "target_sense": 14706 } ]',
        '}',
    ], caption="Source file 2 — the ground truth for the same item.")

    D.listing([
        '// test_dictionary.json  — the gloss lookup',
        '{ "sense_id": 14704, "definition": "\\ar{هرب فلان في الأرض أبعد فيها}" }',
        '{ "sense_id": 14706, "definition": "\\ar{هرب من مسئولياته: تنصل منها، تملص منها}" }',
    ], caption="Source file 3 — the glosses, stored once and shared across all sentences that use them.",
        arabic_lines={1, 2})

    D.p("The two candidate glosses are semantically distant rather than near-synonymous. Sense 14704 is "
        "the literal reading — to flee across the land — while 14706 is figurative: to evade one's "
        "responsibilities. The sentence describes escaping a party by sleeping, which is evasion, so the "
        "figurative sense is correct.")

    # ---- Stage 1: the joined record ---------------------------------
    D.p("**Stage 1 — the joined training record.** The staging script resolves each identifier against the "
        "dictionary and emits one flat JSON object per line.")
    D.listing([
        '{',
        '  "instruction": "You are tasked with performing Word Sense Disambiguation',
        '                  (WSD). Your job is to analyze the given sentence and',
        '                  identify the correct sense for the target word based on',
        '                  the context. For each sense, you are provided with a',
        '                  Sense ID and its definition. Using the context of the',
        '                  sentence, choose the most appropriate sense definition',
        '                  and provide the corresponding Sense ID.",',
        '',
        '  "input": "Sentence: \':-\\ar{هرب من الحفلة بالنوم.}\'',
        '            Target Word: \'\\ar{هرب}\'',
        '            Possible Senses:',
        '            [Sense ID: 14704, Definition: \\ar{هرب فلان في الأرض أبعد فيها}],',
        '            [Sense ID: 14706, Definition: \\ar{هرب من مسئولياته: تنصل منها}]",',
        '',
        '  "output": "14706"',
        '}',
    ], caption="Stage 1 — one line of fine_tuning_dataset_elrazzaz.jsonl. The instruction field is "
               "byte-identical in all 9,952 records; only input and output vary.",
        arabic_lines={9, 10, 12, 13})

    # ---- Stage 2: the rendered prompt --------------------------------
    D.p("**Stage 2 — the rendered training sequence.** The record is wrapped in the Alpaca template. This "
        "string, and nothing else, is what the tokenizer sees.")
    D.listing([
        'Below is an instruction that describes a task, paired with an input that',
        'provides further context. Write a response that appropriately completes',
        'the request.',
        '',
        '### Instruction:',
        'You are tasked with performing Word Sense Disambiguation (WSD). ...',
        '',
        '### Input:',
        'Sentence: \':-\\ar{هرب من الحفلة بالنوم.}\'',
        'Target Word: \'\\ar{هرب}\'',
        'Possible Senses:',
        '[Sense ID: 14704, Definition: \\ar{هرب فلان في الأرض أبعد فيها}], [Sense ID:',
        '14706, Definition: \\ar{هرب من مسئولياته: تنصل منها، تملص منها}]',
        '',
        '### Response:',
        '14706<eos>          <-- present at TRAINING, empty at INFERENCE',
    ], caption="Stage 2 — the rendered sequence. At inference the response section is left empty and the "
               "model generates the continuation.",
        arabic_lines={8, 9, 11, 12})

    D.keypoint("**Both candidate definitions are printed in the prompt.** The model is not recalling what "
               "\\ar{هرب} means; it is choosing between two strings already in front of it. This single "
               "observation underpins the interpretation offered in the results chapter.")

    # ---- Stage 3: generation and extraction --------------------------
    D.p("**Stage 3 — generation, extraction and validation.** The model generates greedily; two regular "
        "expressions recover a digit string; the result is checked against the candidate set.")
    D.listing([
        'raw generation      :  "14706<eos>"',
        'regex 1             :  ^\\s*(\\d+)\\s*<eos>      -> matches, captures "14706"',
        'candidate set       :  {14704, 14706}',
        'validation          :  14706 in candidate set   -> ACCEPT',
        'recorded prediction :  "14706"',
        '',
        '(had neither regex matched, or had the identifier not been offered,',
        ' the prediction recorded would be the string "none")',
    ], caption="Stage 3 — from free-form text to a validated sense identifier.")

    # ---- Stage 4: scoring --------------------------------------------
    D.p("**Stage 4 — scoring.** The prediction is paired with the gold label by list position and appended "
        "to the two flat label vectors from which all metrics are computed.")
    D.listing([
        'y_true[0] = "14706"       (from test_truth.json)',
        'y_pred[0] = "14706"       (from predictions_gemma2_2b.json)',
        '                          -> agreement, contributes 1/3110 to accuracy',
    ], caption="Stage 4 — positional alignment and scoring.")

    D.p("For this particular item the fine-tuned model answered correctly. The significance of that fact "
        "is taken up in Section 7.5.6, because the replicated study reports the same item as a failure "
        "case for a model four times larger.")

    # ---- The split -----------------------------------------------------
    D.h3("6.2.6  How the data is divided")
    D.p("Two independent partitions are involved, and they are easily confused. The first is the "
        "dataset-level split defined by the replicated study; the second is an internal split created "
        "inside the training routine purely for monitoring.")
    D.listing([
        'Dataset A  (15,549 senses over 5,347 unique words)',
        '',
        '   published 64 / 16 / 20 split',
        '   |',
        '   +-- train80   9,952 instances  --+',
        '   |                                |  internal 90/10 shuffle (seed 42)',
        '   |                                +-- 8,956  used for gradient updates',
        '   |                                +--   996  held out for monitoring only',
        '   |',
        '   +-- dev20     2,487 instances      (not used in this work)',
        '   |',
        '   +-- test      3,110 instances      NEVER SEEN IN TRAINING',
        '                                      all reported accuracy comes from here',
    ], caption="How Dataset A is divided. The 90/10 split is internal to training and is not a test set.")
    D.p("The distinction matters. The 996 held-out examples were used to plot the monitoring curve in "
        "Section 6.5.4 and for nothing else; no number reported anywhere in this work derives from them. "
        "The 2,487-instance development split defined by the replicated study was not used at all, since "
        "no hyperparameter search was performed — the published configuration was adopted directly.")
    D.figure("fig_splits.png",
        "Division of Dataset A into published splits and the internal monitoring partition.")


"""Chapter 4 expansions — inserted at the end of §4.1 through §4.5."""


# ---------------------------------------------------------------- 4.1 --
def expand_4_1(D):
    D.h3("4.1.1  Activations, and why the total is worse still")
    D.p("Table 4.1 accounts only for parameter-dependent state. A forward pass also stores intermediate "
        "activations at every layer, because the backward pass needs them to compute gradients. "
        "Activation memory scales with the batch size, the sequence length, the hidden dimension and the "
        "number of layers, and for long sequences it can rival the parameter state.")
    D.p("Two mitigations exist. **Gradient checkpointing** stores activations at only a subset of layers "
        "and recomputes the rest during the backward pass, trading roughly 30% additional compute for a "
        "large memory saving; it is enabled in this work. **Gradient accumulation** processes several "
        "small batches before applying an update, so the effective batch size is decoupled from the "
        "memory a single forward pass requires — the reason the configuration in Chapter 6 uses a "
        "per-device batch of one with eight accumulation steps.")
    D.p("The practical consequence is summarised below, taking a conservative estimate of activation "
        "overhead for short sequences.")
    D.table([
        ["Approach", "Parameter state", "Fits 16 GB?", "Fits 24 GB?"],
        ["Full fine-tuning, fp32 Adam", "≈ 41.7 GB", "No", "No"],
        ["Full fine-tuning, 8-bit optimizer", "≈ 26 GB", "No", "No"],
        ["LoRA, fp16 frozen base", "≈ 6 GB", "Yes", "Yes"],
        ["**QLoRA, 4-bit frozen base**", "**≈ 2.5 GB**", "**Yes, comfortably**", "**Yes**"],
    ], widths=[5.6, 3.0, 2.8, 2.6], align_right={1})
    D.caption("Feasibility of each approach for Gemma 2-2B on the two GPUs relevant to this work.")
    D.p("The second row is worth noting: even replacing the optimizer with an 8-bit implementation, which "
        "is the single largest saving available without freezing anything, leaves full fine-tuning out of "
        "reach. Freezing the base is not one option among several — it is the only one.")


# ---------------------------------------------------------------- 4.2 --
def expand_4_2(D):
    D.h3("4.2.1  How each method works")
    D.p("The table above compresses each method to a single line. Because the choice of LoRA is a claim "
        "that the alternatives are worse for this application, the mechanisms are set out here.")

    D.p("**Adapter layers.** Small trainable modules are inserted after the attention and feed-forward "
        "sub-layers of each block. Each is a bottleneck: a down-projection to a small dimension, a "
        "non-linearity, an up-projection back, and a residual connection:")
    D.eq("h ← h + W_up · σ( W_down · h )")
    D.p("With the bottleneck dimension far below the hidden size, the parameter cost is small. The "
        "difficulty is structural rather than numerical: the adapter is a genuine additional layer, so "
        "every forward pass must traverse it. This increases the sequential depth of the network and adds "
        "latency that cannot be removed after training, which matters for any deployed system.")

    D.p("**Prefix tuning.** Rather than modifying weights, a set of trainable vectors is prepended to the "
        "key and value sequences at every attention layer. The model attends to these virtual positions as "
        "though they were real tokens, so they steer behaviour without touching a single pretrained "
        "parameter. **Prompt tuning** is the simpler variant that prepends learned vectors only at the "
        "input embedding layer.")
    D.p("Both share a defect that is particularly costly here. The virtual tokens occupy positions in the "
        "attention computation, so they consume part of the context window permanently. Given the Arabic "
        "tokenizer fertility measured in Section 2.3 — where an Arabic word already costs 1.79 times what "
        "an English word costs — spending further context on virtual tokens is an unattractive trade. Both "
        "methods are also reported to be sensitive to initialisation and harder to optimise reliably.")

    D.p("**BitFit** trains only the bias terms of the network, freezing every weight matrix. The parameter "
        "count is remarkably small and inference is unaffected. It is inapplicable here for a concrete "
        "reason: as noted in Section 6.4.2, **Gemma 2's linear projections carry no bias terms at all**, "
        "so there is nothing for BitFit to train.")

    D.p("**(IA)³** learns three vectors per layer that rescale the keys, the values, and the intermediate "
        "feed-forward activations by element-wise multiplication. The parameter count is lower than LoRA "
        "by an order of magnitude and the rescaling can be folded into adjacent weights, so inference cost "
        "is minimal. Its expressivity is correspondingly limited: a per-dimension scaling cannot represent "
        "an arbitrary low-rank update, only a diagonal one.")

    D.h3("4.2.2  Why LoRA was selected")
    D.p("Three properties decide it for this application.")
    D.bullet("**Mergeability.** The update is a linear term added to an existing matrix, so after training "
             "it folds into the base weights. The deployed model is architecturally identical to the "
             "original — no extra layers, no added latency, no consumed context. Of the methods above only "
             "(IA)³ shares this, and at much lower expressivity.")
    D.bullet("**Composition with quantization.** Because the adapter is a separate small pair of matrices "
             "held at full precision, the base can be quantized aggressively while the trainable part "
             "remains numerically well-conditioned. This is exactly what QLoRA exploits, and it is not "
             "available to methods that modify the pretrained weights in place.")
    D.bullet("**Comparability.** The replicated study used LoRA. Choosing a different method would "
             "confound the parameter-count question this work is asking with a change of adaptation "
             "method, and the comparison in Chapter 7 would not be interpretable.")
    D.p("The cost of the choice is the rank constraint, developed in Section 4.3.5, which limits how much "
        "the update can express. For learning an output format that limit is not binding; for injecting "
        "knowledge it may be.")


# ---------------------------------------------------------------- 4.3 --
def expand_4_3(D):
    D.h3("4.3.6  Where to place the adapters")
    D.p("Equation 4.2 applies to a single weight matrix, but a decoder layer contains seven. Which of them "
        "to adapt is a design decision with a direct memory cost.")
    D.p("The original LoRA work adapted only the query and value projections, reporting that this "
        "recovered most of the benefit at a fraction of the parameters. Subsequent practice has moved "
        "towards adapting every linear projection, on the reasoning that restricting adaptation to "
        "attention restricts it to how information is routed, leaving the feed-forward layers — where "
        "the interpretability literature localises stored knowledge — untouched.")
    D.p("The configuration used here adapts all seven, following the replicated study. The distribution of "
        "parameters across them is uneven and worth stating explicitly, because it determines what the "
        "adapter is capable of changing.")
    D.table([
        ["Module group", "Modules", "Parameters / layer", "Share"],
        ["Attention", "q, k, v, o_proj", "491,520", "31%"],
        ["**Feed-forward (MLP)**", "gate, up, down_proj", "**1,105,920**", "**69%**"],
        ["Total", "all seven", "1,597,440", "100%"],
    ], widths=[4.0, 3.6, 3.2, 2.2], align_right={2, 3})
    D.caption("Distribution of adapter capacity between attention and feed-forward projections.")
    D.p("More than two thirds of the adapter's capacity sits in the feed-forward block. This is a useful "
        "fact to carry forward: it means the configuration used for supervised fine-tuning already places "
        "most of its capacity where knowledge is stored, and that the limitation for knowledge injection "
        "is therefore the **rank**, not the placement.")

    D.figure("fig_adapter_split.png",
             "Adapter parameters per layer by module. The three feed-forward projections "
             "account for 69% of the adapter's capacity.", width_cm=12.0)

    D.h3("4.3.7  Choosing the rank")
    D.p("The rank r is the one hyperparameter genuinely specific to LoRA. It trades capacity against "
        "memory linearly: doubling r doubles the trainable parameters and, approximately, the optimizer "
        "state.")
    D.p("Published guidance clusters into three regimes. Ranks of 4 to 16 suffice for adapting style, "
        "tone, or an output convention. Ranks of 32 to 64 are typical for general task adaptation. Ranks "
        "of 128 and above are used when the objective is to absorb new domain knowledge rather than new "
        "behaviour. The QLoRA authors additionally report that performance is relatively insensitive to r "
        "beyond a low threshold, provided the adapters are attached to all linear layers.")
    D.p("This work uses r = 32, the value published by the replicated study. Retuning it would improve "
        "neither comparability nor, on the evidence of Chapter 7, accuracy: the task turns out to be a "
        "binary discrimination between two supplied strings, which is a low-dimensional thing to learn. A "
        "rank ablation is recorded in the future work as the cheapest remaining experiment.")

    D.h3("4.3.8  Merging, and serving many adapters")
    D.p("After training, the adapted weight can be materialised once:")
    D.eq("W′ = W₀ + (α / r) · B · A")
    D.p("The result is an ordinary weight matrix of the original shape. Two consequences follow that "
        "matter in practice.")
    D.p("First, **inference cost is exactly that of the base model** — no additional matrix "
        "multiplications, no extra layers, no framework dependency at serving time. This is the property "
        "that distinguishes LoRA from adapter layers.")
    D.p("Second, because the adapter is small and separable, **many task-specific adapters can share one "
        "base model**. A 41.5-megabyte adapter can be swapped, versioned, or distributed independently of "
        "the 2.6-billion-parameter model it modifies. In this project the adapter is the entire trained "
        "artifact: the base weights are downloaded unchanged from a public repository, and everything the "
        "training run produced is contained in a file small enough to attach to an email.")
    D.p("The same separability is what makes the two experiments in this report cleanly comparable. The "
        "supervised fine-tuning adapter and the continued-pretraining adapter are distinct artifacts over "
        "an identical frozen base, so any measured difference between them is attributable to the training "
        "objective and not to drift in the underlying model.")


# ---------------------------------------------------------------- 4.4 --
def expand_4_4(D):
    D.h3("4.4.1  Numerical formats")
    D.p("Before quantization can be discussed, the formats involved should be distinguished. A "
        "floating-point number allocates its bits between a sign, an exponent controlling dynamic range, "
        "and a mantissa controlling precision.")
    D.table([
        ["Format", "Bits", "Exponent / mantissa", "Character"],
        ["fp32", "32", "8 / 23", "Full precision; the reference"],
        ["fp16", "16", "5 / 10", "Half the memory, but a narrow exponent range that overflows easily"],
        ["bf16", "16", "8 / 7", "Same range as fp32, less precision; more robust for training"],
        ["int8", "8", "integer + scale", "Requires a scale factor per tensor or block"],
        ["**NF4**", "**4**", "**non-uniform levels**", "**16 levels placed at normal quantiles**"],
    ], widths=[2.2, 1.4, 3.4, 6.4])
    D.caption("Numerical formats relevant to quantized training.")
    D.p("The distinction between fp16 and bf16 explains a configuration choice recorded in Chapter 6. Both "
        "occupy sixteen bits, but bf16 preserves the exponent range of fp32 at the cost of mantissa bits. "
        "For a model such as Gemma 2, whose logit soft-capping (Section 2.2) makes the forward pass "
        "sensitive to numerical range, the wider exponent is the safer choice, and the run reported here "
        "used bf16.")

    D.h3("4.4.2  Symmetric and asymmetric mapping")
    D.p("Quantization maps a continuous range onto a finite integer grid. Two schemes are common.")
    D.p("**Symmetric** quantization assumes the value distribution is centred on zero and uses a single "
        "scale factor. A value x is encoded as round(x / s) and recovered as ŝ · x_int, where the scale s "
        "is derived from the largest absolute value in the block. Zero maps exactly to zero, which is "
        "convenient and preserves sparsity.")
    D.p("**Asymmetric** quantization adds a zero-point offset, allowing the representable range to be "
        "shifted. It fits skewed distributions better — activations after a ReLU, for instance, are "
        "non-negative — at the cost of an extra stored parameter and an extra operation.")
    D.p("Pretrained weight distributions are approximately zero-centred, so weight quantization is "
        "normally symmetric. This is the setting relevant here.")

    D.h3("4.4.3  Quantization error")
    D.p("Every quantized value carries a rounding error bounded by half the spacing between adjacent "
        "levels. With b bits the grid holds 2^b levels, so halving the bit-width doubles the spacing and "
        "therefore doubles the worst-case error. Moving from 16-bit to 4-bit does not degrade quality by a "
        "factor of four; it reduces the number of representable levels from 65,536 to **16**.")
    D.p("That such aggressive compression is viable at all rests on two observations. Neural network "
        "weights are highly redundant, so small perturbations are absorbed. And the error is random rather "
        "than systematic, so across a matrix multiplication involving thousands of terms the errors "
        "partially cancel.")
    D.p("The errors do not cancel when they are not random, which is the subject of the outlier problem "
        "below.")

    D.h3("4.4.4  Block-wise quantization and the outlier problem")
    D.p("Applying one scale factor to an entire tensor is fragile. The scale is set by the largest "
        "magnitude present, so a single extreme value compresses every other weight into a small part of "
        "the available grid. If one weight is a hundred times larger than the rest, the remaining weights "
        "are confined to roughly one percent of the range — for a 4-bit grid of 16 levels, that means "
        "almost all of them collapse onto the same level.")
    D.p("The remedy is to quantize in **blocks**, typically 64 elements, each with its own scale. An "
        "outlier then damages only its own block, and the other blocks retain full resolution. The cost is "
        "the storage of one scale per block, which is what double quantization (Section 4.5.2) "
        "subsequently addresses.")
    D.p("This is not a theoretical precaution. Dettmers et al. showed that transformers beyond a certain "
        "scale develop **emergent outlier features**: specific hidden dimensions whose activation "
        "magnitudes are orders of magnitude above the rest, appearing consistently across layers and "
        "inputs rather than sporadically. Their emergence is abrupt with model size, and once present, "
        "naive uniform quantization degrades the model sharply. Block-wise schemes exist precisely to "
        "contain this.")

    D.h3("4.4.5  When quantization is applied")
    D.p("Two strategies exist, and the distinction determines what is possible in a project of this size.")
    D.p("**Post-training quantization** takes a trained model and compresses it, optionally using a small "
        "calibration set to choose scale factors well. It is cheap — minutes rather than GPU-days — and "
        "requires no access to the training pipeline. Methods such as GPTQ, which quantizes weights "
        "layer by layer while compensating for the error introduced so far, and AWQ, which protects the "
        "weight channels most important to activations, are refinements of this idea.")
    D.p("**Quantization-aware training** simulates quantization during training so the model learns "
        "weights robust to it. It generally yields better results at very low bit-widths, but requires the "
        "full training budget, which defeats the purpose here.")
    D.keypoint("QLoRA occupies a third position that is neither of these. The base model is quantized "
               "**post-training and then frozen**; it is never updated, so it never needs to be "
               "re-quantized. Learning happens entirely in full-precision adapters sitting alongside it. "
               "This sidesteps the usual trade-off: the memory benefit of aggressive post-training "
               "quantization is obtained without the quality cost of training through a quantized "
               "parameter.")


# ---------------------------------------------------------------- 4.5 --
def expand_4_5(D):
    D.h3("4.5.5  Where the memory actually goes")
    D.p("The headline figure of Table 4.5 is worth decomposing, because the distribution of the remaining "
        "2.5 GB explains why the method scales the way it does.")
    D.table([
        ["Component", "Precision", "Approx. size", "Trainable?"],
        ["Base model weights (2.61B parameters)", "NF4 + double quant", "≈ 1.45 GB", "No — frozen"],
        ["Quantization scales", "8-bit, blocks of 256", "≈ 0.04 GB", "No"],
        ["LoRA adapters (41.5M parameters)", "bf16", "≈ 0.08 GB", "**Yes**"],
        ["Adapter gradients", "bf16", "≈ 0.08 GB", "—"],
        ["Adapter optimizer state (8-bit Adam)", "int8 × 2", "≈ 0.08 GB", "—"],
        ["Activations (batch 1, ~220 tokens, checkpointed)", "bf16", "≈ 0.3–0.8 GB", "—"],
    ], widths=[6.4, 3.0, 2.4, 2.4])
    D.caption("Approximate memory breakdown for QLoRA fine-tuning of Gemma 2-2B.")
    D.keypoint("The frozen base accounts for roughly **60%** of the footprint and everything associated "
               "with learning accounts for under **10%**. This inverts the situation in Table 4.1, where "
               "optimizer state dominated. Once the base is frozen and quantized, the model is essentially "
               "a fixed cost and the trainable machinery is nearly free.")
    D.p("It also explains the scaling behaviour. Doubling the rank doubles three of the small rows and "
        "leaves the largest untouched, so rank is cheap. Doubling the sequence length increases only the "
        "activation row, but quadratically in the attention component — which is why the 4,096-token "
        "sequences of Dataset B were prohibitive while the 216-token average of Dataset A was not.")

    D.h3("4.5.6  How gradients reach the adapters")
    D.p("A question that arises naturally: if the base model is quantized and frozen, how does a gradient "
        "computed at the output reach an adapter attached in the middle of the network?")
    D.p("The answer is that freezing and traversing are different things. The frozen base receives no "
        "gradient **with respect to its own parameters** — those are marked as not requiring gradients, so "
        "no update is accumulated for them. But backpropagation must still compute the gradient **with "
        "respect to the activations** flowing through each layer, because that is the only route by which "
        "the error signal reaches earlier layers. The adapters sit inside the network, not on top of it, "
        "so the chain rule passes through the quantized weights to reach them.")
    D.p("Concretely, each matrix multiplication involving a quantized weight is performed by dequantizing "
        "the block to bf16, multiplying, and discarding the dequantized copy. The backward pass "
        "dequantizes again to compute the activation gradient. The high-precision weight therefore exists "
        "only transiently, inside a single operation, and is never stored.")
    D.p("This is also the source of the throughput cost. Dequantization is arithmetic performed on every "
        "forward and backward pass over every quantized layer, and it is not free.")

    D.h3("4.5.7  The cost side of the trade")
    D.p("QLoRA is frequently described as making training cheaper. It makes training **possible**, which "
        "is different, and the distinction should be stated honestly.")
    D.table([
        ["Dimension", "Effect of QLoRA versus fp16 LoRA"],
        ["Peak memory", "Reduced several-fold — the reason the method is used"],
        ["Throughput", "**Reduced** — dequantization on every forward and backward pass"],
        ["Model quality", "Reported to be close to 16-bit full fine-tuning; not verified here"],
        ["Numerical stability", "Requires care; the compute dtype and attention backend both matter"],
    ], widths=[4.0, 8.0])
    D.caption("What QLoRA costs as well as what it saves.")
    D.p("The quality claim is the one to treat with most care. The QLoRA authors report that 4-bit "
        "adaptation matches 16-bit full fine-tuning across their evaluations, but this work did not run "
        "the 16-bit comparison — it could not, on the available hardware — so no independent confirmation "
        "is offered. What can be said is narrower and is the substance of Chapter 7: the resulting model "
        "matches published results obtained with larger models and 16-bit LoRA, which is consistent with "
        "the claim without establishing it.")


"""Expansions for Chapters 1, 2 and 3."""


# ---------------------------------------------------------------- 1.1 --
def expand_1_1(D):
    D.h3("1.1.1  What unresolved ambiguity costs downstream")
    D.p("The argument that word sense disambiguation matters is easier to make concretely than in the "
        "abstract, because the failure propagates into every system built on top of the text.")
    D.bullet("**Machine translation.** A translator that cannot tell \\ar{نفس} meaning *soul* from "
             "\\ar{نفس} meaning *same* produces fluent output with the wrong content — the most damaging "
             "kind of translation error, because nothing in the result signals that it is wrong.")
    D.bullet("**Search and retrieval.** A query for one sense returns documents using the other. Recall "
             "and precision degrade simultaneously: relevant documents are missed while irrelevant ones "
             "rank highly.")
    D.bullet("**Information extraction.** Systems that populate databases from text propagate the wrong "
             "sense into structured records, where the error becomes permanent and is no longer visible "
             "as a language-processing failure.")
    D.bullet("**Text-to-speech.** Because the ambiguity is created by missing vowels, a synthesiser must "
             "resolve the sense in order to pronounce the word at all. Here disambiguation is not an "
             "enabling step but a hard prerequisite.")
    D.p("This last case is the clearest statement of the problem. In a language that writes its vowels, a "
        "speech synthesiser can read a word without understanding it. In Arabic it cannot: the written "
        "form underdetermines the pronunciation, and choosing between the readings requires exactly the "
        "inference that word sense disambiguation performs.")

    D.h3("1.1.2  Why model size is the practical obstacle")
    D.p("The published results that motivate this work were obtained with models of seven to nine billion "
        "parameters, adapted on datacentre hardware. That is a reasonable research setting and an "
        "unreasonable one for most of the people who would use the result.")
    D.p("The gap is not incremental. As Chapter 4 quantifies, conventionally fine-tuning even a "
        "two-billion-parameter model requires roughly forty-two gigabytes of training state, against the "
        "sixteen gigabytes of a free-tier GPU. Without parameter-efficient methods the entire line of work "
        "is closed to a student, a small research group, or an organisation in a region where such "
        "hardware is not readily available — which, for a language spoken predominantly in exactly those "
        "regions, is a substantive problem rather than a convenience one.")
    D.p("This gives the project a second motivation alongside the scientific one. Establishing that a "
        "two-billion-parameter model suffices is not only a claim about model capacity; it is a claim "
        "about who can participate in Arabic language technology.")


# ---------------------------------------------------------------- 2.1 --
def expand_2_1(D):
    D.h3("2.1.4  Position information")
    D.p("Attention as defined in Equation 2.2 is permutation-invariant: reordering the input rows "
        "reorders the output rows identically, but changes nothing about which token attends to which. "
        "The mechanism has no notion of sequence order, yet order is obviously essential to meaning.")
    D.p("Order is therefore injected separately. The original Transformer added fixed sinusoidal vectors "
        "to the input embeddings. Gemma 2, in common with most current models, instead uses **rotary "
        "position embeddings (RoPE)**, which rotate the query and key vectors by an angle proportional to "
        "their absolute position before the dot product is taken. The rotation is constructed so that the "
        "resulting attention score depends on the *relative* offset between two positions rather than "
        "their absolute indices.")
    D.p("The property that matters for this work is that relative encoding degrades gracefully. A model "
        "that has learned how tokens five positions apart relate can apply that knowledge at any point in "
        "a sequence, which is what allows a fixed-length training regime to generalise across the varying "
        "sentence lengths of a real dataset.")

    D.h3("2.1.5  Residual connections and normalisation")
    D.p("Attention does not operate alone. Each sub-layer in a Transformer block is wrapped in a residual "
        "connection, so that its output is added to its input rather than replacing it:")
    D.eq("h ← h + SubLayer( Norm(h) )")
    D.p("Two consequences follow. The addition gives gradients a direct path from the loss back to every "
        "earlier layer, which is what makes networks of this depth trainable at all. And because each "
        "sub-layer *adds to* a running representation rather than overwriting it, the block can learn a "
        "small refinement instead of an entire transformation.")
    D.keypoint("This is the same structural idea LoRA exploits at the level of individual weight matrices "
               "(Section 4.3.2): keep the existing function and learn an additive correction to it. The "
               "residual stream makes the architecture receptive to that kind of intervention.")
    D.p("Normalisation stabilises the scale of activations before each sub-layer. Gemma 2 uses **RMSNorm**, "
        "which rescales by the root mean square of the activation vector without subtracting a mean. It is "
        "cheaper than standard layer normalisation and empirically as effective. RMSNorm layers are not "
        "linear projections, which is why they fall outside the set of modules a LoRA configuration "
        "targets — a detail that becomes relevant when the adapter placement is enumerated in "
        "Section 4.3.6.")


# ---------------------------------------------------------------- 2.2 --
def expand_2_2(D):
    D.h3("2.2.1  Structure of a decoder block")
    D.p("The configuration in Table 2.1 lists dimensions; this section describes what the 26 layers "
        "actually compute. Each block applies two sub-layers in sequence, each wrapped in the residual "
        "and normalisation pattern of Section 2.1.5.")
    D.listing([
        'for each of the 26 decoder layers:',
        '',
        '    h  <-  h + Attention( RMSNorm(h) )        # 4 projections: q, k, v, o',
        '    h  <-  h + FeedForward( RMSNorm(h) )      # 3 projections: gate, up, down',
        '',
        'final:  logits = Embedding^T . RMSNorm(h)     # tied with input embedding',
    ], caption="Computation performed by each Gemma 2 decoder block. The seven linear projections named "
               "here are exactly the modules the LoRA configuration targets.")
    D.p("The **feed-forward** sub-layer is where the parameter count concentrates. Gemma 2 uses a gated "
        "formulation: the input is projected twice, to `gate` and `up`, one branch is passed through an "
        "activation function, the two are multiplied element-wise, and the result is projected back down "
        "through `down`. With an intermediate size of 9,216 against a hidden size of 2,304, each "
        "feed-forward block is four times wider than the residual stream it operates on.")
    D.p("That width is the reason the three feed-forward projections dominate the adapter parameter count "
        "computed in Section 4.3.4, and it is consistent with the interpretability literature's finding "
        "that feed-forward layers act as the network's key–value memory.")

    D.h3("2.2.2  Attention window")
    D.p("Gemma 2 alternates between local and global attention across its layers: some attend over the "
        "full sequence, others only within a sliding window of recent tokens. The purpose is efficiency at "
        "long context, since full attention costs time and memory quadratic in sequence length while "
        "windowed attention is linear.")
    D.p("For the sequences used in this work the distinction is immaterial. As Section 6.3 reports, the "
        "longest formatted training example is 750 tokens and the median is 211, so every example sits "
        "well inside any window the architecture defines. The mechanism is noted for completeness and "
        "because it is one of the reasons long-sequence work on the alternative benchmark would have been "
        "more tractable than a naive quadratic estimate suggests — though still beyond the memory "
        "available here.")


# ---------------------------------------------------------------- 3.1 --
def expand_3_1(D):
    D.h3("3.1.1  The templatic system in more detail")
    D.p("The interaction between root, pattern and clitic is worth setting out, because it explains both "
        "why Arabic vocabulary is unbounded and why subword tokenizers trained on other languages handle "
        "it badly.")
    D.p("An Arabic word is built by interleaving a **root** — usually three consonants carrying a broad "
        "semantic field — with a **pattern** of vowels and affixes that determines the grammatical "
        "category and the specific sense. The root \\ar{ك-ت-ب} relates to writing; applying different "
        "patterns yields the forms in Table 3.1. Crucially the consonants remain in order and the pattern "
        "is interposed *between* them, rather than being prefixed or suffixed as in English derivation.")
    D.p("Two properties follow. The system is **productive**: a competent speaker can form and understand "
        "a pattern applied to a root they have never seen combined, so no fixed vocabulary list can be "
        "complete. And it is **non-concatenative**: the morphemes are interleaved rather than "
        "concatenated, so segmenting a word into meaningful units cannot be done by splitting the string "
        "at boundaries.")
    D.keypoint("Subword tokenizers are concatenative by construction — they split strings into contiguous "
               "pieces. Applied to a non-concatenative morphology, the pieces they produce need not align "
               "with morphemes at all. This is the structural reason behind the fertility measurement in "
               "Section 2.3, and it is why the penalty is a property of the writing system rather than a "
               "deficiency that more training data would remove.")
    D.p("A further complication is the **broken plural**, in which a noun is pluralised by changing its "
        "internal vowel pattern rather than by adding a suffix: \\ar{كتاب} (book) pluralises to "
        "\\ar{كتب} (books). The singular and plural share no affix, and a system that expects "
        "pluralisation to be suffixal will treat them as unrelated strings — while, as Table 3.1 shows, "
        "\\ar{كتب} is simultaneously the undiacritised form of two distinct verbs.")


# ---------------------------------------------------------------- 3.5 --
def expand_3_5(D):
    D.h3("3.5.4  The Lesk algorithm")
    D.p("One knowledge-based method deserves description rather than a single line, because it is used as "
        "a baseline in Chapter 7 and the interpretation of that baseline depends on knowing exactly what "
        "was computed.")
    D.p("The Lesk algorithm rests on a simple hypothesis: the correct sense of an ambiguous word is the "
        "one whose dictionary definition shares the most vocabulary with the surrounding context. The "
        "original formulation compared the glosses of *all* words in the context against one another, "
        "which is combinatorially expensive. The **simplified** variant, used here, compares the context "
        "directly against each candidate gloss:")
    D.eq("sense = argmax over c in C of  | tokens(context) ∩ tokens(gloss(c)) |")
    D.p("Ties are broken by a fixed rule, and the choice of rule matters on a binary dataset — Chapter 7 "
        "reports the baseline under both conventions, because the difference between them turns out to be "
        "larger than the contribution of the overlap signal itself.")
    D.p("Simplified Lesk is a useful baseline for three reasons. It requires **no training data**, so it "
        "can be computed on any dataset that supplies glosses. It uses the **same information** the "
        "generative model is given — the context and the candidate definitions — so the comparison is "
        "fair rather than a comparison against a weaker input. And it isolates a specific hypothesis: "
        "that the task is solvable by surface lexical matching. If a neural model substantially "
        "outperforms it, that difference is attributable to something beyond string overlap.")
    D.p("Its known weaknesses are equally relevant. Dictionary glosses are short, so the overlap statistic "
        "is computed over very few tokens and is correspondingly noisy. It is sensitive to preprocessing, "
        "since morphological variants of the same word will not match as strings unless normalised. And "
        "it has no notion of semantic similarity: two paraphrases sharing no vocabulary score zero "
        "overlap. Section 7.3.2 shows that on this dataset these weaknesses are decisive.")


"""Round-two expansions: Chapters 2, 3 and 5 to their page targets."""


# ---------------------------------------------------------------- 2.3 --
def expand_2_3(D):
    D.h3("2.3.5  What the fragmentation looks like")
    D.p("The aggregate fertility figure conceals something more interesting than a simple penalty. "
        "Applying the Gemma 2 tokenizer to individual Arabic words and inspecting the pieces it produces "
        "shows that the segmentation is not merely *more* fragmented than English — it is fragmented in "
        "ways that bear no relation to the structure of the language.")
    D.table([
        ["Arabic word", "Tokens", "Pieces produced", "English gloss", "Tokens"],
        ["\\ar{كتاب}  (book)", "1", "\\ar{كتاب}", "book", "1"],
        ["\\ar{وبكتابهم}  (and with their book)", "3", "\\ar{وب} | \\ar{كتاب} | \\ar{هم}",
         "and with their book", "4"],
        ["\\ar{مكتبة}  (library)", "3", "\\ar{م} | \\ar{كت} | \\ar{بة}", "library", "1"],
        ["\\ar{نفس}  (soul)", "2", "\\ar{ن} | \\ar{فس}", "soul", "1"],
        ["\\ar{هرب}  (escaped)", "2", "\\ar{ه} | \\ar{رب}", "escaped", "1"],
        ["\\ar{مسئولياته}  (his responsibilities)", "5",
         "\\ar{مس} | \\ar{ئ} | \\ar{ول} | \\ar{يات} | \\ar{ه}", "his responsibilities", "3"],
    ], widths=[4.4, 1.6, 4.0, 3.6, 1.4], align_right={1, 4})
    D.caption("Gemma 2 tokenization of individual Arabic words, with their English glosses for comparison.")
    D.p("Three patterns are visible, and they complicate the simple story that Arabic is uniformly "
        "penalised.")
    D.p("**Frequent whole words survive.** \\ar{كتاب} (book) is a single token, exactly as its English "
        "gloss is. Where a form appeared often enough in the pretraining corpus, the merge operations "
        "preserved it intact.")
    D.p("**Occasionally the split is morphologically correct.** The four-morpheme word \\ar{وبكتابهم} "
        "decomposes into \\ar{وب} (the proclitic cluster), \\ar{كتاب} (the stem), and \\ar{هم} (the "
        "possessive enclitic) — a segmentation a linguist would endorse, and one that costs fewer tokens "
        "than the four-word English rendering. This is the best case.")
    D.keypoint("**But the general case is arbitrary.** \\ar{مكتبة} (library), formed from the same root as "
               "\\ar{كتاب}, splits into \\ar{م} | \\ar{كت} | \\ar{بة} — pieces that correspond to no "
               "morpheme and destroy the visible relationship to the root. \\ar{هرب}, the very verb "
               "examined in the worked example of Chapter 6, is split as \\ar{ه} | \\ar{رب}, severing the "
               "root itself.")
    D.p("The reason follows from Section 3.1: byte-pair encoding is a concatenative procedure, splitting "
        "strings into contiguous substrings, while Arabic morphology is non-concatenative, interleaving "
        "root consonants with a vocalic pattern. A contiguous split cannot in general recover an "
        "interleaved structure. Where the tokenizer appears to segment correctly it is because a frequent "
        "surface string happened to coincide with a morpheme boundary, not because the algorithm has any "
        "notion of one.")
    D.p("For this project the consequence is bounded but real. The model must learn to recognise that "
        "\\ar{ه} | \\ar{رب} and other fragments of the same root refer to related concepts, without any "
        "structural hint that they do. That it succeeds — reaching 90.42% on a task that turns on exactly "
        "such distinctions — is evidence that a sufficiently trained model can compensate for a tokenizer "
        "poorly matched to its input, but it should not be mistaken for the tokenizer being adequate.")


# ---------------------------------------------------------------- 2.4 --
def expand_2_4(D):
    D.h3("2.4.1  Exposure bias")
    D.p("Teacher forcing creates a mismatch between training and use that is worth naming. During "
        "training, every prediction is conditioned on a **correct** prefix drawn from the data. During "
        "generation, each prediction is conditioned on the model's **own** previous outputs. A model that "
        "has only ever seen correct prefixes has no experience of recovering from its own mistakes, so an "
        "early error can compound across a long generation.")
    D.p("This phenomenon — exposure bias — is a serious concern for open-ended generation of paragraphs or "
        "documents. It is a minor one here. The output required in this work is a single sense identifier "
        "of a few tokens followed by an end-of-sequence marker, so there is essentially no opportunity for "
        "an error to compound. The empirical result in Section 7.5.1, that the model produced zero "
        "malformed outputs across 3,110 generations, is consistent with that expectation.")

    D.h3("2.4.2  One objective, two uses")
    D.p("Equation 2.6 is the objective for both experiments in this report, which can obscure how "
        "differently it is being applied. The difference lies entirely in what text the tokens come from.")
    D.table([
        ["", "Continued pretraining", "Instruction fine-tuning"],
        ["Sequence content", "Raw domain text", "Instruction, input, and the desired response"],
        ["What is predicted", "The next token of natural text",
         "The next token of a structured response"],
        ["What the model learns", "The distribution of the domain",
         "A mapping from a prompt format to an answer format"],
    ], widths=[3.2, 4.4, 4.8])
    D.caption("The same cross-entropy objective applied to two different kinds of sequence.")
    D.p("The loss function cannot distinguish these cases; it minimises surprise over whatever tokens it "
        "is shown. The distinction is imposed entirely by the choice of training data and, in the "
        "instruction case, by the optional decision to mask the prompt. This is why Chapter 4 treats the "
        "two as separate objectives despite their shared mathematics, and why the dissociation drawn later "
        "is a claim about data and effect rather than about loss functions.")


# ---------------------------------------------------------------- 2.5 --
def expand_2_5(D):
    D.h3("2.5.1  Comparing the two families on this task")
    D.p("The trade-offs are easier to weigh side by side.")
    D.table([
        ["Property", "Bidirectional encoder", "Decoder-only generative model"],
        ["Context available to the target", "Both sides of the word", "Preceding text only"],
        ["Output mechanism", "Classification head over a fixed label set",
         "Free-text generation, parsed afterwards"],
        ["Handling a new sense inventory", "Head must be resized and retrained",
         "None — the inventory is supplied in the prompt"],
        ["Failure modes", "Confident wrong class", "Refusal, hallucination, malformed output"],
        ["Typical size for this task", "0.1 – 0.4B parameters", "2 – 70B parameters"],
        ["Adaptation cost", "Low", "High without parameter-efficient methods"],
    ], widths=[3.6, 4.2, 4.6])
    D.caption("Encoder and decoder approaches to word sense disambiguation compared.")
    D.p("The encoder has genuine structural advantages: bidirectional context is strictly more information, "
        "and a classification head cannot emit an invalid answer. The strongest published encoder result "
        "on this dataset — 76.68% F1, cited in Section 3.5.3 — nonetheless sits well below the fine-tuned "
        "generative results, which suggests that on this benchmark the advantages of scale and pretraining "
        "breadth outweigh the architectural ones.")
    D.p("The decoder's distinctive weakness is that its output is unconstrained. A classification head "
        "returns a label by construction; a language model returns text that must be interpreted, and can "
        "return text that means nothing. The replicated study's 638 refusals from a single model "
        "illustrate the cost. Chapter 7 reports that after fine-tuning this failure mode vanished "
        "entirely, which is the empirical answer to the objection — but it is an answer that must be "
        "measured rather than assumed, which is why the extraction pipeline of Section 6.6.2 records "
        "invalid outputs as a distinct category rather than folding them into ordinary errors.")


# ---------------------------------------------------------------- 3.3 --
def expand_3_3(D):
    D.h3("3.3.1  The normalisation applied in this work")
    D.p("Because normalisation choices affect reported numbers, the exact procedure used in the offline "
        "analyses of Chapter 7 is specified here. It is applied only to string-comparison measurements — "
        "the lexical-overlap baseline and the gloss-similarity statistics — and never to text presented "
        "to the model.")
    D.listing([
        '1. strip diacritics        U+064B .. U+0652, plus U+0670 (dagger alef)',
        '2. strip tatweel           U+0640, the letter-stretching character',
        '3. unify alef variants     أ  إ  آ   ->  ا',
        '4. unify ya                ى          ->  ي',
        '5. unify ta marbuta        ة          ->  ه',
        '6. strip punctuation       all non-word, non-space characters',
        '7. split on whitespace',
        '8. discard single-character tokens',
    ], caption="Normalisation pipeline used for the offline string-comparison analyses.",
        arabic_lines={2, 3, 4})
    D.p("Two of these steps deserve justification. Discarding single-character tokens removes residual "
        "fragments and stray letters that would otherwise inflate overlap counts by matching "
        "coincidentally; the cost is that genuine one-letter particles are lost, but these carry little "
        "disambiguating information. Unifying ta marbuta to ha is the more debatable step, since the two "
        "are distinct letters that happen to be written interchangeably in some conventions; unifying them "
        "increases recall of true matches at the risk of a small number of false ones.")
    D.keypoint("No morphological analyser was used. Two forms of the same root will therefore fail to "
               "match unless their surface strings coincide. Every overlap-based figure in Chapter 7 is "
               "consequently a **conservative lower bound**: a morphologically-aware implementation could "
               "only find more matches, never fewer.")


# ---------------------------------------------------------------- 3.6 --
def expand_3_6(D):
    D.h3("3.6.1  Why Dataset A")
    D.p("Six datasets were available, and the choice among them shapes what any result means. Four "
        "criteria applied.")
    D.bullet("**Public availability.** Datasets A and B are the two distributed openly with documented "
             "splits, which is also why the replicated study selected them. Reproducibility requires that "
             "a reader be able to obtain the data.")
    D.bullet("**Direct comparability.** The central research question is whether a smaller model matches "
             "published results. That requires the *same* dataset and the *same* splits as the published "
             "work; a different benchmark would make the comparison meaningless regardless of the "
             "accuracy obtained.")
    D.bullet("**Compute feasibility.** Dataset B's sequences reach approximately 4,096 tokens. At the "
             "measured Arabic fertility, and given that attention memory grows quadratically in sequence "
             "length, training on it exceeds the available budget by a wide margin. This was a hard "
             "constraint rather than a preference.")
    D.bullet("**Gloss availability.** The generative formulation used here requires that each candidate "
             "sense be accompanied by a natural-language definition that can be placed in the prompt. "
             "Datasets annotated only with sense labels, without glosses, would require a different task "
             "formulation.")
    D.p("Dataset A satisfies all four. It is also, by the replicated study's own analysis, the easier of "
        "the two benchmarks — it is dictionary-derived with few candidates per item, whereas Dataset B is "
        "corpus-derived with many. That asymmetry is acknowledged rather than hidden: the results in "
        "Chapter 7 concern a benchmark whose structure makes the task tractable, and Section 7.7 records "
        "the limits this places on generalisation.")


# ---------------------------------------------------------------- 5.5 --
def expand_5_5(D):
    D.h3("5.5.1  Reproducibility in the surrounding literature")
    D.p("A recurring difficulty in positioning this work is that the published results it compares against "
        "are not fully reproducible, and the gaps are worth recording because they bear on how the "
        "comparison in Chapter 7 should be read.")
    D.bullet("**Per-item predictions are not released.** Only aggregate accuracy and macro-F1 are "
             "published. This makes a paired significance test between this work and the published models "
             "impossible, which is why Section 7.2.1 can report overlapping confidence intervals but not "
             "a formal comparison.")
    D.bullet("**Decoding was not constrained in the zero-shot setting.** The replicated study states that "
             "neither deterministic decoding nor temperature limits were enforced, so those figures are "
             "single samples from a stochastic process. Re-running the same evaluation would not "
             "necessarily reproduce them.")
    D.bullet("**Splits are custom.** Neither dataset ships official partitions, so the study constructed "
             "its own 64/16/20 division. Any work not adopting exactly those splits is not strictly "
             "comparable. This report adopts them unchanged for that reason.")
    D.p("None of this is unusual, and none of it is a criticism of the authors specifically — the same "
        "gaps are near-universal in the surrounding literature. It does mean that comparisons in this area "
        "rest on shared datasets and reported numbers rather than on verifiable per-item agreement, and "
        "claims should be correspondingly cautious.")
    D.p("This work takes two small steps in the other direction. Decoding is greedy and therefore "
        "deterministic, so its figures are exactly reproducible from the saved adapter. And the "
        "per-item predictions, error analyses and baseline computations are retained as files alongside "
        "the report, so the analyses in Chapter 7 can be checked rather than taken on trust.")


"""Chapter 5 top-up: positioning against the encoder line and the scale question."""


def expand_5_2(D):
    D.h3("5.2.1  What the encoder results establish")
    D.p("The encoder line of work matters to this report for a reason beyond historical completeness: it "
        "supplies the only published result on this dataset obtained by a fundamentally different "
        "architecture, and therefore the only check on whether the generative framing is doing real work.")
    D.p("ORCA's 76.68% F1 with AraBERT v2 was obtained with a model of roughly 135 million parameters — "
        "smaller than the models discussed in Section 5.1 by a factor of fifty to seventy, and smaller "
        "than the model used in this report by a factor of nearly twenty. It also had access to "
        "bidirectional context, which the decoder-only models do not.")
    D.table([
        ["Approach", "Model", "Parameters", "Reported"],
        ["Encoder + classification head", "AraBERT v2 (via ORCA)", "≈ 0.14B", "76.68 F1"],
        ["Generative, prompted", "ChatGPT (via GPTAraEval)", "undisclosed", "53.49 F1 (3-shot)"],
        ["Generative, fine-tuned", "Qwen 2.5-7B", "7B", "90.77 acc / 83.98 F1"],
        ["Generative, fine-tuned", "DeepSeek-R1-Q (AraReasoner)", "14B", "86.27 F1"],
        ["**Generative, fine-tuned**", "**Gemma 2-2B (this work)**", "**2B**", "**90.42 acc / 83.33 F1**"],
    ], widths=[4.4, 4.4, 2.4, 3.0])
    D.caption("Published approaches to Dataset A, ordered by architectural family.")
    D.p("Two cautions apply to reading this table. The figures are not uniformly comparable — the encoder "
        "and prompted results are reported as F1 alone, and Section 7.4 shows that the macro-F1 statistic "
        "behaves peculiarly on this dataset, so the F1 column should be treated as indicative rather than "
        "as a ranking. And the splits are not guaranteed identical across all five rows; only the last "
        "three demonstrably share the partition described in Section 5.1.1.")
    D.keypoint("Read with those caveats, the table still shows something coherent. Prompting a very large "
               "general model underperforms a small task-specific encoder, while fine-tuning a mid-sized "
               "open model beats both. **Adaptation to the task matters more than either architecture or "
               "raw scale** — which is the pattern this report extends downward by another factor of "
               "three and a half.")


def expand_5_3(D):
    D.h3("5.3.1  The unexamined assumption about scale")
    D.p("Across the literature surveyed here, a consistent methodological pattern is visible: models are "
        "selected from the largest tier that the available hardware permits, and no study reports "
        "adaptation below seven billion parameters on this task. The reason is not that smaller models "
        "were tested and found wanting; they simply were not tested.")
    D.p("This is understandable — larger models are the ones producing headline results, and a study "
        "asking whether generative models can perform Arabic WSD naturally reaches for the strongest "
        "candidates. But it leaves an assumption in place without examination. If the fine-tuned results "
        "on Dataset A cluster within 1.4 accuracy points across models spanning seven to nine billion "
        "parameters, as Table 5.1 shows they do, then parameter count is evidently not the operative "
        "variable within that range. Whether the pattern continues below it is an empirical question that "
        "the literature has not asked.")
    D.p("There is also a practical dimension. Arabic is spoken predominantly in regions where access to "
        "datacentre-class accelerators is uneven, so a result that requires a 24-gigabyte card to "
        "reproduce is less useful to the community most affected by it than one that requires a free-tier "
        "notebook. Establishing the lower bound is therefore not only a scientific question about "
        "capacity but a question about who is able to build on the work.")