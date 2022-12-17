# grammar for parsing during evaluation
# for items from Vigliocco & Nicol 1998

from nltk import CFG

english_grammar_VN_98 = CFG.fromstring("""
	Q -> VP S
	S -> DP VP
	
	DP -> D NP
	NP -> N | NP PP
	N -> N_sg | N_pl | ADJ N_sg | ADJ N_pl
	
	VP -> V | V ADJ | V EOS
	PP -> P DP
	
	D -> 'the'
	
	N_sg -> 'advertisement' | 'announcement' | 'article' | 'author' | 'bill' | 'computer'
	N_sg -> 'contract' | 'crowd' | 'deck' | 'discussion' | 'friend' | 'gift' | 'helicopter'
	N_sg -> 'lesson' | 'letter' | 'manual' | 'meal' | 'museum' | 'design' | 'engine'
	N_sg -> 'path' | 'photo' | 'prescription' | 'producer' | 'publisher' | 'statue' | 'support'
	N_sg -> 'switch' | 'telegram' | 'threat' | 'tour' | 'train' | 'truck' | 'bridge'
	N_sg -> 'club' | 'director' | 'writer' | 'speech' | 'accountant' | 'program' | 'actor'
	N_sg -> 'street' | 'yacht' | 'topic' | 'editor' | 'baby' | 'flight' | 'government'
	N_sg -> 'machine' | 'guest' | 'picture' | 'lake' | 'girl' | 'doctor' | 'movie' | 'book'
	N_sg -> 'garden' | 'platform' | 'light' | 'soldier' | 'president' | 'museum' | 'city'
	
	N_pl -> 'advertisements' | 'announcements' | 'articles' | 'authors' | 'bills' | 'computers'
	N_pl -> 'contracts' | 'crowds' | 'decks' | 'discussions' | 'friends' | 'gifts' | 'helicopters'
	N_pl -> 'lessons' | 'letters' | 'manuals' | 'meals' | 'museums' | 'designs' | 'engines'
	N_pl -> 'paths' | 'photos' | 'prescriptions' | 'producers' | 'publishers' | 'statues' | 'supports'
	N_pl -> 'switches' | 'telegrams' | 'threats' | 'tours' | 'trains' | 'trucks' | 'bridges'
	N_pl -> 'clubs' | 'directors' | 'writers' | 'speeches' | 'accountants' | 'programs' | 'actors'
	N_pl -> 'streets' | 'yachts' | 'topics' | 'editors' | 'babies' | 'flights' | 'governments'
	N_pl -> 'machines' | 'guests' | 'pictures' | 'lakes' | 'girls' | 'doctors' | 'movies' | 'books'
	N_pl -> 'gardens' | 'platforms' | 'lights' | 'soldiers' | 'presidents' | 'museums' | 'cities'
	
	V -> 'do' | 'does' | 'seem'
	
	EOS -> '[EOS]'
	
	P -> 'from' | 'by' | 'of' | 'with' | 'for' | 'in' | 'about' | 'around' | 'to' | 'on'
	
	ADJ -> 'colorful' | 'important' | 'confusing' | 'young' | 'reasonable' | 'available'
	ADJ -> 'acceptable' | 'restless' | 'spacious' | 'boring' | 'blonde' | 'expensive' | 'safe'
	ADJ -> 'interesting' | 'emotional' | 'comprehensible' | 'delicious' | 'open' | 'successful'
	ADJ -> 'scenic' | 'blurry' | 'ready' | 'famous' | 'wealthy' | 'impressive' | 'solid'
	ADJ -> 'hidden' | 'unexpected' | 'serious' | 'disappointing' | 'crowded' | 'noisy' | 'new'
""")
setattr(english_grammar_VN_98, 'lang', 'en_VN_98')