# UTC-IE

This is the code for ACL 2023
paper [UTC-IE: A Unified Token-pair Classification Architecture for Information Extraction](https://aclanthology.org/2023.acl-long.226/).

## Requirements
```
python==3.7.13 
fastNLP==1.0.1
transformers==4.18.0                   
fitlog==0.9.13
pytorch==1.11
```

## File structure

You need to put your data in the parallel folder of this repo.

```tree
    - UTC-IE/
        - train_ner.py
        - train_re.py
        ...
    - dataset/
        - conll2003
            - train.txt
            - test.txt
            - dev.txt
        - en-ontonotes
            - ...
        - en_ace04
        - en_ace05
        - genia_w2ner
        - UniRE_ace2005
        - UniRE_SciERC
        - ace05E
        - ace05E+
        - ERE

```

## Prepare data

The data you prepare should be like examples below.

### conll03

The first column is words, the second column is tags. We assume the tag is the BIO-tagging.

```text
LONDON NNP I-NP I-LOC
1996-08-30 CD I-NP O

West NNP I-NP I-MISC
Indian NNP I-NP I-MISC
all-rounder NN I-NP O
Phil NNP I-NP I-PER
```

### en-ontonotes

Our code will automatically convert the data to the conll03 format. The detail of the preprocessing is
in https://github.com/yhcc/OntoNotes-5.0-NER.

```text
bc/msnbc/00/msnbc_0000   0    0            first    RB  (TOP(S(ADVP*           -    -   -    Dan_Abrams   (ORDINAL)  (ARGM-TMP*        *            *        *          -
bc/msnbc/00/msnbc_0000   0    1               up    RB             *           -    -   -    Dan_Abrams          *            *        *            *        *          -
bc/msnbc/00/msnbc_0000   0    2               on    IN          (PP*           -    -   -    Dan_Abrams          *            *        *            *        *          -
```

### en_ace04, en_ace05, genia_w2ner

For nested dataset, each line is a jsonline, contains ners and sentences keys. The preprocess method follows
the [CNN_Nested_NER repo](https://github.com/yhcc/CNN_Nested_NER/tree/master/preprocess).

```text
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
```

### UniRE_ace2005, UniRE_SciERC

For RE dataset, each line is a jsonline, contains ners, relations, clusters and sentences keys.

```text
{"sentences": [["CNN_ENG_20030626_193133", ".8"], ["NEWS", "STORY"], ["2003-06-26", "20:00:18"], ["so", "what", "are", "the", "clintons", ",", "ford", "and", "arnold", "schwarzenegger", "and", "the", "cast", "of", "\"", "friends", "\"", "have", "in", "common", "?"], ["they", "all", "love", "california", "."], ["that", "is", "certainly", "true", "and", "know", "aaron", "tonkin", "and", "according", "to", "to", "\"", "vanity", "fair", ",", "\"", "he", "'s", "at", "the", "heart", "of", "half", "a", "dozen", "investigations", "with", "fraud", "and", "angry", "investor", "lawsuits", "a-plenty", "."], ["joining", "us", "is", "brian", "burrow", "to", "explain", "it", "for", "us", "."], ["who", "this", "guy", "?"], ["garn", "tonkin", "?"], ["brian", ",", "can", "you", "hear", "me", "?"], ["i", "think", "heard", "someone", "say", "something", "."], ["can", "you", "hear", "me", "?"], ["you", "'re", "on", "the", "air", "."], ["one", "of", "the", "great", "moments", "of", "live", "television", ",", "is", "n't", "it", "?"], ["bliian", ",", "can", "you", "hear", "me", "?"], ["all", "right", "."], ["we", "'ll", "take", "a", "--"], ["i", "can", "hear", "you", "now", ",", "anderson", "."], ["well", ",", "maybe", "not", "k.", "you", "hear", "many", "?"], ["just", "barely", "."], ["we", "'ll", "take", "a", "short", "break", "."], ["we", "'ll", "be", "right", "back", "."], ["become", "a", "member", "of", "...", "...", "at", "westin.com", "."], ["jooirx", "joining", "us", ",", "brian", "burrow", "."], ["who", "is", "aaron", "tonkin", "."], ["he", "is", "one", "of", "the", "biggest", "if", "not", "the", "biggest", "names", "in", "the", "world", "of", "hollywood", "fund-raisers", "."], ["a", "person", "who", "would", "put", "together", "celebrities", "with", "charities", "to", "raise", "a", "lot", "of", "money", "."], ["he", "was", "perhaps", "best", "known", "as", "kind", "of", "the", "focal", "point", "between", "bill", "and", "hillary", "clinton", "in", "the", "hollywood", "community", "."], ["what", "'s", "interesting", ",", "i", "mean", ",", "reading", "the", "article", ",", "basically", "a", "lot", "of", "the", "celebrities", "paid", "by", "him", "to", "attend", "charity", "functions", "."], ["yeah", "."], ["that", "'s", "one", "of", "the", "things", "that", "was", "most", "surprising", "to", "me", "was", "in", "learning", "that", "in", "one", "of", "the", "--", "one", "major", "fund-raiser", ",", "for", "instance", ",", "honoring", "former", "president", "clinton", ",", "former", "president", "ford", "paid", "by", "mr", "."], ["tonkin", "and", "sylvester", "stl", "loan", "was", "paid", "."], ["i", "thought", "it", "was", "for", "charity", "."], ["yeah", "."], ["i", "guess", "apparently", "not", "and", "a", "lot", "of", "the", "celebrities", "making", "money", "from", "this", "."], ["are", "they", "reporting", "it", "to", "the", "irs", "."], ["i", "guess", "at", "least", "one", "irs", "investigation", "about", "this", "."], ["right", "now", ",", "aaron", "tonkin", "factors", "in", "no", "fewer", "than", "seven", "different", "federal", "and", "state", "investigations", "."], ["one", "of", "them", "being", "an", "irs", "investigation", "into", "several", "million", "dollars", "worth", "of", "cash", "and", "gift", "that", "is", "he", "gave", "to", "many", "many", "hollywood", "celebrities", "."], ["now", ",", "his", "attorney", "gave", "us", "a", "statement", "i", "'ll", "put", "it", "on", "the", "screen", "."], ["it", "says", "--", "is", "that", "possible", "?"], ["in", "fact", ",", "allege", "ledged", "one", "of", "the", "principle", "reasons", "he", "took", "so", "much", "fun", "from"], ["2003-06-26", "20:04:01"]], "ner": [[], [], [], [[10, 10, "PER"], [12, 12, "PER"], [14, 15, "PER"], [18, 18, "PER"]], [[27, 27, "PER"], [30, 30, "GPE"]], [[38, 39, "PER"], [49, 49, "PER"], [45, 46, "ORG"], [63, 63, "PER"]], [[70, 71, "PER"], [68, 68, "PER"], [76, 76, "PER"]], [[80, 80, "PER"], [78, 78, "PER"]], [[82, 83, "PER"]], [[90, 90, "PER"], [85, 85, "PER"], [88, 88, "PER"]], [[95, 95, "PER"], [92, 92, "PER"]], [[102, 102, "PER"], [100, 100, "PER"]], [[104, 104, "PER"]], [], [[128, 128, "PER"], [123, 123, "PER"], [126, 126, "PER"]], [], [[133, 133, "PER"]], [[141, 141, "PER"], [144, 144, "PER"], [138, 138, "PER"]], [[151, 151, "PER"]], [], [[158, 158, "PER"]], [[165, 165, "PER"]], [[173, 173, "PER"]], [[184, 185, "PER"], [182, 182, "PER"]], [[189, 190, "PER"], [187, 187, "PER"]], [[192, 192, "PER"], [194, 194, "PER"], [208, 208, "PER"], [207, 207, "ORG"], [202, 202, "PER"]], [[211, 211, "PER"], [212, 212, "PER"], [216, 216, "PER"], [218, 218, "ORG"]], [[226, 226, "PER"], [236, 236, "PER"], [244, 244, "ORG"], [238, 238, "PER"], [240, 241, "PER"], [245, 245, "PER"]], [[251, 251, "PER"], [266, 266, "PER"], [263, 263, "PER"]], [], [[312, 312, "PER"], [305, 305, "PER"], [304, 304, "PER"], [309, 309, "PER"], [308, 308, "PER"], [285, 285, "PER"]], [[314, 314, "PER"], [316, 318, "PER"]], [[322, 322, "PER"]], [], [[331, 331, "PER"], [340, 340, "PER"]], [[347, 347, "PER"], [352, 352, "ORG"]], [[354, 354, "PER"], [359, 359, "ORG"]], [[367, 368, "PER"], [378, 378, "ORG"]], [[399, 399, "PER"], [404, 404, "ORG"], [386, 386, "ORG"], [405, 405, "PER"]], [[415, 415, "PER"], [412, 412, "ORG"], [409, 409, "PER"], [410, 410, "PER"]], [], [[440, 440, "PER"]], []], "relations": [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[208, 208, 207, 207, "ORG-AFF"]], [], [[238, 238, 240, 241, "PER-SOC"]], [], [], [], [], [], [], [], [], [], [], [[405, 405, 404, 404, "ORG-AFF"]], [[409, 409, 410, 410, "PER-SOC"]], [], [], []], "clusters": [], "doc_key": "CNN_ENG_20030626_193133.8"}
{"sentences": [["CNN_IP_20030404", ".1600.00", "-2"], ["STORY"], ["2003-04-04T16", ":00:00", "-05:00"], ["Iraq", "Warns", "of", "Unconventional", "Counter", "Attack", "Tonight", ";", "Saddam", "Sends"], ["Message", "Suggesting", "he", "is", "Alive", ",", "Still", "Defiant"], ["BLITZER"], ["Attacking", "on", "the", "ground", "and", "calling", "in", "air", "strikes", ",", "Kurdish", "fighters", "and", "U.S", ".", "troops", "have", "dislodged", "stubborn", "Iraqi", "soldiers", "at", "a", "key", "bridge", "in", "northern", "Iraq", "."], ["Iraqis", "pulled", "back", "on", "the", "road", "to", "Mosul", "as", "CNN", "'s", "Jane", "Arraf", "reports", "from", "the", "battlefield", "."], ["(", "BEGIN", "VIDEOTAPE", ")"], ["JANE", "ARRAF", ",", "CNN", "CORRESPONDENT", "(", "voice-over", ")"], ["A", "few", "miles", "further", "to", "Mosul", ",", "Kurdish", "militia", "and", "their", "flags", "speeding", "to", "a", "town", "abandoned", "by", "Iraqi", "forces", "."], ["After", "a", "day", "of", "fighting", ",", "the", "Iraqis", "were", "driven", "back", "five", "kilometers", ",", "about", "three", "miles", "down", "the", "main", "road", "west", "towards", "Mosul", "from", "the", "Kurdish", "city", "of", "Erbil", "."], ["Soldiers", "were", "Kurdish", ",", "but", "the", "special", "forces", "calling", "in", "air", "strikes", "on", "Iraqi", "positions", "were", "American", "."], ["As", "U.S", ".", "warplanes", "dropped", "bombs", "near", "the", "town", ",", "Kurdish", "fighters", "moved", "forward", "."], ["(", "on", "camera", ")", ":", "There", "'s", "a", "slow-and-steady", "battle", "going", "on", "here", "for", "control", "of", "the", "key", "bridge", "."], ["It", "'s", "a", "bridge", "over", "the", "river", "on", "the", "main", "road", "to", "Mosul", "."], ["The", "Iraqis", "are", "firing", "artillery", "like", "that", "in", "response", "to", "the", "Americans", "are", "calling", "in", "air", "strikes", "."], ["The", "Peshmerga", "are", "just", "down", "the", "road", ",", "and", "the", "Iraqis", "have", "retreated", ",", "but", "they", "'re", "still", "holding", "on", "to", "the", "bridge", "."], ["(", "voice-over", ")", ":", "That", "blast", "turned", "out", "to", "be", "a", "rocket", "-", "propelled", "grenade", ",", "but", "there", "'s", "plenty", "of", "artillery", "and", "mortar", "fire", "to", "come", "."], ["UNIDENTIFIED", "MALE"], ["Are", "you", "coming", "here", "?"], ["ARRAF"], ["Less", "than", "20", "minutes", "later", ",", "with", "Iraqi", "defenses", "pounded", "by", "the", "bombing", ",", "they", "no", "longer", "held", "the", "bridge", "over", "the", "Khoser", "River", "."], ["In", "this", "vehicle", ",", "the", "only", "Iraqi", "casualties", "we", "saw", "."], ["Their", "military", "radio", "and", "guns", "indicating", "they", "were", "combatants", "."], ["(", "on", "camera", ")", ":", "It", "'s", "still", "smoldering", ",", "this", "truck", "with", "three", "Iraqi", "soldiers", "."], ["It", "was", "either", "shelled", "or", "bombed", "."], ["Now", ",", "it", "'s", "a", "small", "part", "of", "this", "battle", "for", "the", "bridge", "just", "behind", "us", "."], ["These", "were", "lying", "on", "the", "ground", "next", "to", "the", "truck", "."], ["Somebody", "picked", "them", "up", "."], ["It", "'s", "an", "I.D.", "card", ",", "presumably", "for", "one", "of", "the", "soldiers", "."], ["The", "only", "thing", "you", "can", "really", "tell", "from", "it", "is", "that", "he", "was", "27", "years", "old", "."], ["And", "a", "snapshot", "from", "when", "his", "life", "was", "still", "ahead", "of", "him", "."], ["Other", "evidence", "of", "a", "hurried", "retreat", "amid", "the", "smoldering", "vehicles", ",", "a", "discarded", "gas", "mask", "."], ["Military", "documents", "."], ["Notebooks", "with", "verses", "from", "the", "Koran", "."], ["Kurdish", "fighters", "planted", "their", "flag", "and", "left", "it", "waving", "in", "the", "smoke", "of", "the", "bombed-out", "Iraqi", "truck", "."], ["The", "village", "of", "Manguba", "(", "ph", ")", ",", "just", "32", "kilometers", ",", "about", "23", "miles", "from", "Mosul", "is", "now", "in", "Kurdish", "hands", "."], ["One", "Kurdish", "soldier", "did", "the", "honors", "of", "tearing", "up", "a", "poster", "of", "Saddam", "Hussein", "."], ["Not", "just", "the", "poster", ",", "but", "the", "frame", "and", "cardboard", "backing", "as", "well", "."], ["They", "'ve", "been", "waiting", "a", "long", "time", "."], ["Jane", "Arraf", ",", "CNN", ",", "on", "the", "bridge", "over", "the", "Khoser", "River", "in", "northern", "Iraq", "."], ["(", "END", "VIDEO", "CLIP", ")"], ["BLITZER"], ["Thanks", "very", "much", "."], ["Judy", ",", "dangerous", "assignment", "for", "all", "of", "our", "reporters", ",", "our", "embedded", "journalists", "."], ["And", "one", "of", "them", ",", "not", "from", "our", "CNN", ",", "but", "from", "a", "very", "close", "friend", "of", "all", "of", "ours", ",", "Michael", "Kelly", ",", "paid", "that", "price", "."], ["The", "first", "American", "embedded", "journalist", "to", "have", "gotten", "killed", "in", "covering", "this", "war", "."], ["We", "'ll", "have", "more", "on", "him", "at", "5:00", "on", "our", "special", "edition", "of", "\"", "WOLF", "BLITZER", "REPORTS", ".", "\""], ["Until", "then", ",", "thanks", ",", "Judy", ",", "back", "to", "you", "."], ["WOODRUFF"], ["Thanks", ",", "Wolf", "."], ["And", "we", "'ll", "be", "watching", "."], ["Michael", "Kelly", ",", "of", "course", ",", "with", "\"", "Atlantic", "Monthly", ",", "\"", "and", "a", "columnist", "for", "the", "\"", "Washington", "Post", ".", "\""], ["Well", ",", "at", "least", "three", "U.S", ".", "service", "members", "were", "killed", "today", "in", "an", "apparent", "suicide", "bomb", "attack", "at", "a", "checkpoint", "near", "the", "Haditah", "Dam", "northwest", "of", "Baghdad", "."], ["Those", "deaths", "are", "not", "yet", "included", "in", "the", "official", "casualty", "count", "."], ["At", "last", "report", ",", "though", ",", "41", "Americans", "had", "been", "killed", "by", "hostile", "forces", ",", "13", "others", "by", "friendly", "fire", "or", "in", "accidents", "."], ["Of", "the", "27", "British", "troops", ",", "in", "all", "who", "have", "been", "killed", ",", "at", "least", "19", "of", "them", "were", "said", "to", "be", "victims", "of", "friendly", "fire", "or", "accidents", "."], ["The", "official", "Iraqi", "casualty", "figures", "remain", "the", "same", "this", "day", ",", "including", "more", "than", "400", "civilians", "killed", ",", "Iraqi", "authorities", "say", "."], ["And", "U.S", ".", "Central", "Command", "says", "more", "than", "4,000", "Iraqis", "have", "been", "captured", ",", "but", "British", "officials", "believe", "that", "figure", "is", "much", "higher", "."], ["We", "know", "that", "seven", "Americans", "are", "known", "to", "be", "held", "as", "prisoner", "of", "war", "in", "Iraq", "."], ["Another", "15", "Americans", "listed", "as", "missing", "in", "action", "."], ["Well", ",", "funeral", "services", "were", "held", "today", "in", "Colorado", "for", "21-year", "-", "old", "Marine", "Corporal", "Randall", "Kent", "Rosacker", "."], ["He", "was", "killed", "in", "action", "in", "Iraq", "."], ["His", "story", "is", "especially", "poignant", "."], ["His", "father", "is", "a", "navy", "commander", "."], ["He", "received", "word", "of", "his", "son", "'s", "death", "within", "hours", "after", "the", "submarine", "that", "he", "was", "assigned", "to", "returned", "home", "from", "an", "extended", "deployment", "."], ["The", "Iraqis", "promise", "an", "unconventional", "attack", "on", "U.S", ".", "troops", "tonight", "."], ["Do", "they", "mean", "guerilla", "warfare", ",", "as", "in", "Vietnam", "?"], ["We", "'re", "going", "to", "ask", "our", "military", "analyst", "."], ["That", "'s", "coming", "up", "."]], "ner": [[], [], [], [], [], [[25, 25, "PER"]], [[45, 45, "GPE"], [36, 36, "PER"], [39, 39, "GPE"], [50, 50, "FAC"], [37, 37, "PER"], [41, 41, "PER"], [46, 46, "PER"], [53, 53, "LOC"]], [[55, 55, "PER"], [66, 67, "PER"], [64, 64, "ORG"], [60, 60, "FAC"], [62, 62, "GPE"]], [], [[77, 78, "PER"], [81, 81, "PER"], [80, 80, "ORG"]], [[100, 100, "GPE"], [104, 104, "PER"], [103, 103, "GPE"], [92, 92, "PER"], [90, 90, "GPE"], [93, 93, "PER"], [95, 95, "PER"]], [[113, 113, "PER"], [133, 133, "GPE"], [135, 135, "GPE"], [132, 132, "PER"], [126, 126, "FAC"], [129, 129, "GPE"]], [[150, 150, "GPE"], [139, 139, "PER"], [137, 137, "PER"], [144, 144, "PER"], [153, 153, "GPE"]], [[163, 163, "GPE"], [165, 165, "PER"], [158, 158, "VEH"], [156, 156, "GPE"], [160, 160, "WEA"], [166, 166, "PER"]], [[188, 188, "FAC"], [182, 182, "LOC"]], [[190, 190, "FAC"], [193, 193, "FAC"], [196, 196, "LOC"], [200, 200, "FAC"], [202, 202, "GPE"]], [[205, 205, "PER"], [215, 215, "PER"], [208, 208, "WEA"]], [[232, 232, "PER"], [237, 237, "PER"], [244, 244, "FAC"], [228, 228, "FAC"], [223, 223, "PER"]], [[267, 267, "WEA"], [260, 260, "WEA"], [257, 257, "WEA"], [269, 269, "WEA"]], [[275, 275, "PER"]], [[277, 277, "PER"], [279, 279, "LOC"]], [[281, 281, "PER"]], [[289, 289, "GPE"], [296, 296, "PER"], [301, 301, "FAC"], [304, 305, "LOC"], [290, 290, "PER"]], [[313, 313, "GPE"], [309, 309, "VEH"], [315, 315, "PER"], [314, 314, "PER"]], [[318, 318, "PER"], [324, 324, "PER"], [326, 326, "PER"], [319, 319, "ORG"], [322, 322, "WEA"]], [[342, 342, "GPE"], [333, 333, "VEH"], [339, 339, "VEH"], [343, 343, "PER"]], [[345, 345, "VEH"]], [[364, 364, "FAC"], [354, 354, "VEH"], [367, 367, "PER"]], [[378, 378, "VEH"]], [[380, 380, "PER"]], [[396, 396, "PER"], [393, 393, "PER"]], [[409, 409, "PER"], [401, 401, "PER"]], [[420, 420, "PER"], [426, 426, "PER"]], [[437, 437, "VEH"]], [[444, 444, "ORG"]], [], [[469, 469, "GPE"], [454, 454, "PER"], [470, 470, "VEH"], [455, 455, "PER"], [457, 457, "PER"]], [[492, 492, "PER"], [473, 473, "GPE"], [475, 475, "GPE"], [488, 488, "GPE"]], [[496, 496, "PER"], [497, 497, "PER"], [507, 508, "PER"]], [], [[524, 524, "PER"]], [[539, 539, "FAC"], [542, 543, "LOC"], [532, 533, "PER"], [535, 535, "ORG"], [546, 546, "LOC"]], [], [[553, 553, "PER"]], [], [[563, 563, "PER"], [566, 566, "PER"], [570, 570, "PER"], [558, 558, "PER"], [565, 565, "ORG"], [568, 568, "ORG"]], [[575, 575, "PER"], [587, 587, "PER"], [593, 594, "PER"], [573, 573, "PER"], [580, 580, "ORG"], [579, 579, "ORG"], [591, 591, "ORG"], [589, 589, "PER"]], [[602, 602, "GPE"], [604, 604, "PER"]], [[619, 619, "PER"], [614, 614, "ORG"], [623, 623, "ORG"]], [[642, 642, "PER"], [638, 638, "PER"]], [[644, 644, "PER"]], [[647, 647, "PER"]], [[650, 650, "PER"]], [[655, 656, "PER"], [669, 669, "PER"], [663, 664, "ORG"], [673, 674, "ORG"]], [[682, 682, "GPE"], [685, 685, "PER"], [697, 697, "FAC"], [700, 701, "FAC"], [704, 704, "GPE"], [693, 693, "WEA"], [684, 684, "ORG"]], [], [[725, 725, "PER"], [734, 734, "PER"], [731, 731, "PER"]], [[746, 746, "PER"], [750, 750, "PER"], [759, 759, "PER"], [745, 745, "GPE"], [757, 757, "PER"], [764, 764, "PER"]], [[773, 773, "GPE"], [789, 789, "GPE"], [786, 786, "PER"], [790, 790, "PER"]], [[794, 794, "GPE"], [808, 808, "GPE"], [796, 797, "ORG"], [802, 802, "PER"], [809, 809, "PER"]], [[832, 832, "GPE"], [828, 828, "PER"], [821, 821, "PER"], [817, 817, "ORG"]], [[836, 836, "PER"]], [[851, 851, "GPE"], [858, 860, "PER"], [857, 857, "PER"], [856, 856, "ORG"]], [[868, 868, "GPE"], [862, 862, "PER"]], [[870, 870, "PER"]], [[876, 876, "PER"], [877, 877, "PER"], [881, 881, "PER"], [880, 880, "ORG"]], [[888, 888, "PER"], [883, 883, "PER"], [887, 887, "PER"], [897, 897, "PER"], [895, 895, "VEH"], [896, 896, "VEH"], [902, 902, "LOC"]], [[909, 909, "GPE"], [915, 915, "GPE"], [917, 917, "PER"]], [[921, 921, "GPE"], [928, 928, "GPE"]], [[936, 936, "ORG"], [937, 937, "PER"], [930, 930, "ORG"], [935, 935, "ORG"]], []], "relations": [[], [], [], [], [], [], [[36, 36, 37, 37, "GEN-AFF"], [41, 41, 39, 39, "ORG-AFF"], [46, 46, 45, 45, "ORG-AFF"], [50, 50, 53, 53, "PART-WHOLE"], [46, 46, 50, 50, "PHYS"]], [[66, 67, 64, 64, "ORG-AFF"]], [], [[81, 81, 80, 80, "ORG-AFF"]], [[92, 92, 93, 93, "ORG-AFF"], [104, 104, 103, 103, "ORG-AFF"], [104, 104, 100, 100, "PHYS"]], [[132, 132, 133, 133, "GEN-AFF"], [113, 113, 126, 126, "PHYS"]], [[144, 144, 153, 153, "ORG-AFF"]], [[156, 156, 158, 158, "ART"], [165, 165, 166, 166, "GEN-AFF"]], [], [[196, 196, 200, 200, "PART-WHOLE"], [193, 193, 196, 196, "PHYS"]], [[205, 205, 208, 208, "ART"]], [], [[257, 257, 260, 260, "PART-WHOLE"]], [], [[277, 277, 279, 279, "PHYS"]], [], [[301, 301, 304, 305, "PHYS"], [290, 290, 289, 289, "ORG-AFF"]], [[314, 314, 309, 309, "ART"], [314, 314, 313, 313, "ORG-AFF"]], [[318, 318, 322, 322, "ART"]], [[343, 343, 339, 339, "ART"], [343, 343, 342, 342, "ORG-AFF"]], [], [[367, 367, 364, 364, "PHYS"]], [], [], [], [], [], [], [], [], [[454, 454, 455, 455, "GEN-AFF"], [469, 469, 470, 470, "ART"]], [[473, 473, 488, 488, "PHYS"]], [[497, 497, 496, 496, "GEN-AFF"]], [], [], [[532, 533, 535, 535, "ORG-AFF"], [539, 539, 542, 543, "PHYS"], [532, 533, 539, 539, "PHYS"], [542, 543, 546, 546, "PART-WHOLE"]], [], [], [], [[566, 566, 565, 565, "ORG-AFF"], [570, 570, 568, 568, "ORG-AFF"]], [[587, 587, 589, 589, "PER-SOC"]], [[604, 604, 602, 602, "GEN-AFF"]], [], [], [], [], [], [[655, 656, 663, 664, "ORG-AFF"], [669, 669, 673, 674, "ORG-AFF"]], [[684, 684, 682, 682, "PART-WHOLE"], [685, 685, 684, 684, "ORG-AFF"], [697, 697, 700, 701, "PHYS"], [700, 701, 704, 704, "PHYS"], [685, 685, 697, 697, "PHYS"]], [], [], [[746, 746, 745, 745, "ORG-AFF"]], [[790, 790, 789, 789, "ORG-AFF"]], [[796, 797, 794, 794, "PART-WHOLE"], [809, 809, 808, 808, "ORG-AFF"]], [[821, 821, 832, 832, "PHYS"]], [], [[857, 857, 856, 856, "ORG-AFF"]], [[862, 862, 868, 868, "PHYS"]], [], [[876, 876, 877, 877, "PER-SOC"], [881, 881, 880, 880, "ORG-AFF"]], [[897, 897, 895, 895, "ART"], [888, 888, 887, 887, "PER-SOC"]], [[917, 917, 915, 915, "ORG-AFF"]], [], [[937, 937, 935, 935, "ORG-AFF"]], []], "clusters": [], "doc_key": "CNN_IP_20030404.1600.00-2"}
```

### ace05E, ace05E+, ERE

ace05E: [DyGIE++](https://github.com/dwadden/dygiepp/tree/master/scripts/data/ace-event) data format to
the format used by OneIE. 

ace05E+: can downloaded from [Paper: Zero-shot Event Trigger and Argument Classification](https://github.com/CogComp/ZEC)

ERE: converts raw ERE datasets (LDC2015E29, LDC2015E68, LDC2015E78) to the format used by OneIE. 

```text
{"doc_id": "CNN_CF_20030303.1900.02", "sent_id": "CNN_CF_20030303.1900.02-0", "entity_mentions": [], "relation_mentions": [], "event_mentions": [], "tokens": ["CNN_CF_20030303.1900.02"], "sentence": "CNN_CF_20030303.1900.02"}
{"doc_id": "CNN_CF_20030303.1900.02", "sent_id": "CNN_CF_20030303.1900.02-1", "entity_mentions": [], "relation_mentions": [], "event_mentions": [], "tokens": ["STORY"], "sentence": "STORY"}
```

## How to run

### NER

```shell
# dconll2003
CUDA_VISIBLE_DEVICES=1 python train_ner.py --lr 3e-5 -b 4 -n 30 -d dconll2003 --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 0 -a 3 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.1 --use_size_embed 2

# dontonotes
CUDA_VISIBLE_DEVICES=2 python train_ner.py --lr 1e-5 -b 4 -n 10 -d dontonotes --cross_depth 1 --cross_dim 64 --use_s2 1 --use_gelu 0 -a 3 --use_ln 2 --biaffine_size 300 --drop_s1_p 0 --attn_dropout 0.15 --warmup 0.2 --use_size_embed 1

# ace2004
CUDA_VISIBLE_DEVICES=3 python train_ner.py --lr 2e-5 -b 48 -n 50 -d ace2004 --cross_depth 2 --cross_dim 64 --use_s2 0 --use_gelu 1 -a 1 --use_ln 2 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.2 --warmup 0.1

# ace2005
CUDA_VISIBLE_DEVICES=4 python train_ner.py --lr 3e-5 -b 6 -n 50 -d ace2005 --cross_depth 3 --cross_dim 64 --use_s2 1 --use_gelu 0 -a 8 --use_ln 0 --biaffine_size 200 --drop_s1_p 0 --attn_dropout 0.2 --warmup 0.1

# genia
CUDA_VISIBLE_DEVICES=5 python train_ner.py --lr 7e-6 -b 8 -n 5 -d genia --cross_depth 2 --cross_dim 32 --use_s2 1 --use_gelu 1 -a 1 --use_ln 0 --biaffine_size 100 --drop_s1_p 0 --attn_dropout 0.2 --warmup 0.1
```

### RE

```shell
# ace2005
CUDA_VISIBLE_DEVICES=6 python train_re.py -d ace2005 --cross_depth 3 --cross_dim 200 --use_ln 2 --empty_rel_weight 0.1 --symmetric 1 -b 32 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200

# ace2005/albert
CUDA_VISIBLE_DEVICES=7 python train_re.py -d ace2005 --model_name albert-xxlarge-v1 --cross_depth 3 --cross_dim 200 --use_ln 2 --empty_rel_weight 0.1 --symmetric 1 -b 32 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200

# sci
CUDA_VISIBLE_DEVICES=1 python train_re.py -d sciere --cross_depth 4 --cross_dim 200 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 32 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200
```

### Symmetric RE

```shell
# ace2005(sym+context)
CUDA_VISIBLE_DEVICES=2 python train_sym_re.py -d race2005_ --cross_depth 3 --cross_dim 200 --use_ln 2 --empty_rel_weight 0.1 --symmetric 1 -b 32 -n 70 --lr 3e-5 --use_s2 1 --biaffine_size 300 --use_size_embed 1 --use_sym_rel 2

# sci(sym+context)
CUDA_VISIBLE_DEVICES=3 python train_sym_re.py -d sciere_ --cross_depth 3 --cross_dim 100 --use_ln 0 --empty_rel_weight 1 --symmetric 1 -b 16 -n 100 --lr 3e-5 --use_s2 1 --biaffine_size 200 --use_sym_rel 1
```

### EE

```shell
# ACE05E
CUDA_VISIBLE_DEVICES=1 python train_ee.py --lr 7e-6 --cross_dim 200 --use_ln 1 --biaffine_size 300 --drop_s1_p 0 -b 12 --model_name microsoft/deberta-v3-large -a 1 -n 100 --cross_depth 3 -d ace05E

# ACE05E+
CUDA_VISIBLE_DEVICES=3 python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0.1 -b 32 --model_name microsoft/deberta-v3-large -a 1 -n 100 --cross_depth 3 -d ace05E+
  
# ere
CUDA_VISIBLE_DEVICES=4 python train_ee.py --lr 1e-5 --cross_dim 150 --use_ln 1 --biaffine_size 300 --drop_s1_p 0 -b 32 --model_name microsoft/deberta-v3-large -a 1 --attn_dropout 0.2 -n 70 --cross_depth 2 -d ere
```

### Joint IE

```shell
# ACE05E+-joint
CUDA_VISIBLE_DEVICES=5 python train_ie.py -d ace05E+ --cross_dim 200 --lr 1e-5 -b 48 -n 100 --cross_depth 3 --use_ln 0 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300

# ere-joint
CUDA_VISIBLE_DEVICES=6 python train_ie.py -d ere --cross_dim 200 --lr 1e-5 -b 4 -n 100 --cross_depth 3 --use_ln 2 --drop_s1_p 0 --empty_rel_weight 0.1 --use_gelu 1 --biaffine_size 300
```