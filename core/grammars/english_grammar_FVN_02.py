# grammar for parsing during evaluation
# for items from Franck, Vigliocco & Nicol 2002

from nltk import CFG

english_grammar_FVN_02 = CFG.fromstring("""
	S -> DP VP
	
	DP -> D NP
	NP -> N | NP PP
	N -> N_sg | N_pl | Comp Comp N_sg | Comp Comp N_pl
	N -> ADJ N_sg | ADJ N_pl
	
	VP -> V ADJ
	PP -> P DP
	
	D -> 'the' | 'my'
	
	Comp -> 'real' | 'estate'
	
	N_sg -> 'advertisement' | 'office' | 'agent' | 'announcement' | 'director' | 'foundation' 
	N_sg -> 'article' | 'writer' | 'magazine' | 'author' | 'speech' | 'city' | 'computer' | 'cousin' | 'program'
	N_sg -> 'experiment' | 'contract' | 'actor' | 'film' | 'discussion' | 'topic' | 'paper' | 'dog'
	N_sg -> 'path' | 'lake' | 'friend' | 'editor' | 'gift' | 'daughter' | 'visitor' | 'helicopter'
	N_sg -> 'flight' | 'canyon' | 'lesson' | 'government' | 'country' | 'manual' | 'developer' | 'machine'
	N_sg -> 'mast' | 'deck' | 'yacht' | 'meal' | 'guest' | 'inn-keeper' | 'museum' | 'picture' | 'poet'
	N_sg -> 'design' | 'engine' | 'plane' | 'payment' | 'service' | 'school' | 'photo' | 'girl' | 'baby'
	N_sg -> 'post' | 'support' | 'platform' | 'prescription' | 'doctor' | 'clinic' | 'producer' | 'movie'
	N_sg -> 'artist' | 'publisher' | 'book' | 'king' | 'setting' | 'astronomer' | 'statue' | 'garden'
	N_sg -> 'mansion' | 'switch' | 'light' | 'stairway' | 'telegram' | 'soldier' | 'threat' | 'president'
	N_sg -> 'company' | 'tour' | 'monument' | 'train' | 'truck' | 'bridge' | 'stream'    
	
	N_pl -> 'advertisements' | 'offices' | 'agents' | 'announcements' | 'directors'
	N_pl -> 'foundations' | 'articles' | 'writers' | 'magazines' | 'authors' | 'speeches' | 'cities'
	N_pl -> 'computers' | 'cousins' | 'programs' | 'experiments' | 'contracts' | 'actors' | 'films' | 'discussions'
	N_pl -> 'topics' | 'papers' | 'dogs' | 'paths' | 'lakes' | 'friends' | 'editors' | 'gifts'
	N_pl -> 'daughters' | 'visitors' | 'helicopters' | 'flights' | 'canyons' | 'lessons' | 'governments'
	N_pl -> 'countries' | 'manuals' | 'developers' | 'machines' | 'masts' | 'decks' | 'yachts' | 'meals'
	N_pl -> 'guests' | 'inn-keepers' | 'museums' | 'pictures' | 'poets' | 'designs' | 'engines'
	N_pl -> 'planes' | 'payments' | 'services' | 'schools' | 'photos' | 'girls' | 'babies' | 'posts'
	N_pl -> 'supports' | 'platforms' | 'prescriptions' | 'doctors' | 'clinics' | 'producers' | 'movies'
	N_pl -> 'artists' | 'publishers' | 'books' | 'kings' | 'settings' | 'astronomers' | 'statues'
	N_pl -> 'gardens' | 'mansions' | 'switches' | 'lights' | 'stairways' | 'telegrams' | 'soldiers'
	N_pl -> 'threats' | 'presidents' | 'companies' | 'tours' | 'monuments' | 'trains' | 'trucks'
	N_pl -> 'bridges' | 'streams'
	
	V -> 'seem' | 'seems'
	
	P -> 'from' | 'of' | 'by' | 'for' | 'about' | 'with' | 'in' | 'on' | 'around' | 'over' | 'to' | 'near' 
	
	ADJ -> 'colorful' | 'important' | 'confusing' | 'young' | 'available' | 'acceptable' | 'boring'
	ADJ -> 'happy' | 'blonde' | 'expensive' | 'safe' | 'interesting' | 'emotional' | 'comprehensible'
	ADJ -> 'sturdy' | 'delicious' | 'open' | 'successful' | 'blurry' | 'solid' | 'ready' | 'famous'
	ADJ -> 'wealthy' | 'appropriate' | 'impressive' | 'hidden' | 'unexpected' | 'serious' | 'disappointing' 
	ADJ -> 'crowded' | 'noisy' | 'new'
""")
setattr(english_grammar_FVN_02, 'lang', 'en_FVN_02')