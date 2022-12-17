# code adapted from Bob Frank's grammars.py
import re
import random

from nltk import Tree, PCFG, CFG
from nltk import nonterminals

from typing import *
from .generator import generate, format_tree_string
from .generator import create_dataset_json, combine_dataset_jsons
from .generator import grep_next_subtree

from .keybaseddefaultdict import KeyBasedDefaultDict

# this creates a dictionary that returns a default string for sg and pl
# based on the value of the key passed to it
# override for specific verbs that display non-default behavior as below
PAST_PRES = {
	'sg': KeyBasedDefaultDict(lambda s: re.sub(r'ed$', 's', s)),
	'pl': KeyBasedDefaultDict(lambda s: re.sub(r'ed$', '', s)),
}

PAST_PRES['sg'].update({
	k: re.sub(r'ed$', 'es', k) for k in [
		'liked', 
		'inspired',
		'faced',
		'predated',
		'eclipsed',
		'shaded',
	]
})

PAST_PRES['pl'].update({
	k: re.sub(r'ed$', 'e', k) for k in [
		'liked', 
		'inspired',
		'faced',
		'predated',
		'eclipsed',
		'shaded',
	]
})

english_grammar_RC_PP = PCFG.fromstring("""
	S -> DP VP [1.0]
	
	DP -> D NP [1.0]
	NP -> N [0.8] | NP PP [0.1] | NP CP [0.1]
	N -> N_sg [0.5] | N_pl [0.5]
	
	VP -> V DP [1.0]
	
	PP -> P DP [1.0]
	CP -> C VP [1.0]
	
	D -> 'the' [1.0]
	
	N_sg -> 'student' [0.1]  | 'professor' [0.1]  | 'headmaster' [0.1]  | 'friend' [0.1]
	N_sg -> 'assistant' [0.1] | 'dean' [0.1] | 'advisor' [0.1] | 'colleague' [0.1]
	N_sg -> 'president' [0.1] | 'chancellor' [0.1]
	
	N_pl -> 'students' [0.1] | 'professors' [0.1] | 'headmasters' [0.1] | 'friends' [0.1] 
	N_pl -> 'assistants' [0.1] | 'deans' [0.1] | 'advisors' [0.1] | 'colleagues' [0.1] 
	N_pl -> 'presidents' [0.1] | 'chancellors' [0.1]
	
	V -> 'helped' [0.1] | 'visited' [0.1] | 'liked' [0.1] 
	V -> 'bothered' [0.1] | 'inspired' [0.1] | 'recruited' [0.1] 
	V -> 'assisted' [0.1] | 'confounded' [0.1] | 'accosted' [0.1] | 'avoided' [0.1]
	
	P -> 'of' [0.2] | 'near' [0.2] | 'by' [0.2] | 'behind' [0.2] | 'with' [0.2]
	
	C -> 'that' [1.0]
""")
setattr(english_grammar_RC_PP, 'lang', 'en_RC_PP')

english_grammar_RC_PP_gen = PCFG.fromstring("""
	S -> DP VP [1.0]
	
	DP -> D NP [1.0]
	NP -> N [0.8] | NP PP [0.1] | NP CP [0.1]
	N -> N_sg [0.5] | N_pl [0.5]
	
	VP -> V DP [1.0]
	
	PP -> P DP [1.0]
	CP -> C VP [1.0]
	
	D -> 'the' [1.0]
	
	N_sg -> 'house' [0.1]  | 'shed' [0.1]  | 'tree' [0.1]  | 'machine' [0.1]
	N_sg -> 'building' [0.1]  | 'church' [0.1]  | 'office' [0.1]  | 'motel' [0.1]
	N_sg -> 'barn' [0.1]  | 'shack' [0.1]
		
	N_pl -> 'houses' [0.1] | 'sheds' [0.1] | 'trees' [0.1] | 'machines' [0.1] 
	N_pl -> 'buildings' [0.1] | 'churches' [0.1] | 'offices' [0.1] | 'motels' [0.1] 
	N_pl -> 'barns' [0.1] | 'shacks' [0.1]
	
	V -> 'blocked' [0.1] | 'dwarfed' [0.1] | 'overlooked' [0.1] 
	V -> 'faced' [0.1] | 'predated' [0.1] | 'eclipsed' [0.1] 
	V -> 'concealed' [0.1] | 'shaded' [0.1] | 'overshadowed' [0.1] | 'adjoined' [0.1]
	
	P -> 'beside' [0.2] | 'near' [0.2] | 'by' [0.2] | 'behind' [0.2] | 'around' [0.2]
	
	C -> 'that' [1.0]
""")
setattr(english_grammar_RC_PP_gen, 'lang', 'en_RC_PP_gen')

def present_pair(grammar: PCFG) -> Tuple:
	past_tree = generate(grammar)
	pres_tree = reinflect(past_tree)
	return past_tree, 'pres', pres_tree

def past_pair(grammar: PCFG) -> Tuple:
	past_tree = generate(grammar)
	return past_tree, 'past', past_tree

def reinflect(t: Tree) -> Tree:
	# Make a deep copy so we don't mess up the original tree
	t_copy = t.copy(deep=True)
	
	# get the main clause verb
	main_clause_VP = grep_next_subtree(t_copy, r'^VP$')
	main_clause_V = grep_next_subtree(main_clause_VP, r'^V$')
	
	# get the number of the main clause subject
	main_clause_subject = grep_next_subtree(t_copy, r'^DP$')
	main_clause_subject = grep_next_subtree(main_clause_subject, r'^NP$')
	while grep_next_subtree(main_clause_subject[0], r'^NP$'):
		main_clause_subject = grep_next_subtree(main_clause_subject[0], r'^NP$')
	
	main_clause_subject = grep_next_subtree(main_clause_subject, r'^N_')
	subject_number = 'sg' if str(main_clause_subject.label()).endswith('sg') else 'pl'
	
	# map the past form of the verb to the present form based on the number of the subject
	main_clause_V[0] = PAST_PRES[subject_number][main_clause_V[0]]
	
	return t_copy

def pres_or_past(grammar: PCFG, pres_p: float = 0.5) -> Tuple:
	return present_pair(grammar) if random.random() < pres_p else past_pair(grammar)

def pres_or_past_no_pres_dist(grammar: PCFG, pres_p: float = 0.5) -> Tuple:
	source, pfx, target = pres_or_past(grammar, pres_p)
	
	# for English, we do not care about distractors in the past tense, since they do not affect attraction
	# in fact, we WANT some of these for training
	if pfx == 'pres':
		# otherwise, we need to modify the tree to change the number of all interveners to match the subject's number
		main_clause_subject = grep_next_subtree(source, r'^DP$')
		
		# this works now because the main clause subject is always the first noun!
		# it will need to be changed if we add nouns before the main clause subject
		pre_verb_noun_positions = [
			pos 
			for pos in main_clause_subject.treepositions() 
			if 	not isinstance(main_clause_subject[pos],str) and 
				re.search(r'^N_', str(main_clause_subject[pos].label()))
		]
		main_clause_subject_pos = pre_verb_noun_positions[0]
		
		if len(pre_verb_noun_positions) > 1:
			main_clause_subject_number = re.findall(r'_(.*)', str(main_clause_subject[main_clause_subject_pos].label()))[0]
			intervener_positions = [
				pos 
				for pos in pre_verb_noun_positions[1:] 
					if not re.findall(r'_(.*)', str(main_clause_subject[pos].label()))[0] == main_clause_subject_number
			]
			
			for t in [source, target]:
				
				main_clause_subject = grep_next_subtree(t, r'^DP$')
				
				for pos in intervener_positions:
					if main_clause_subject_number == 'sg':
						main_clause_subject[pos] = Tree(
							main_clause_subject[main_clause_subject_pos].label(),
							[re.sub('s$', '', main_clause_subject[pos][0])]
						)
					elif main_clause_subject_number == 'pl':
						if not main_clause_subject[pos][0].endswith('s'):
							main_clause_subject[pos] = Tree(
								main_clause_subject[main_clause_subject_pos].label(),
								[f'{main_clause_subject[pos][0]}s']
							)
			
	return source, pfx, target

def test_file(grammar: PCFG = english_grammar_RC_PP, n: int = 10, filename: str = 'test.txt') -> None:
	"""
	Create a small test file with n pairs of formatted present and past tense sentences
	"""
	s = [present_pair(grammar) for _ in range(n)]
	s = [(format_tree_string(past_tree), pfx, format_tree_string(present_tree)) for past_tree, pfx, present_tree in s]
	with open(filename, 'w') as out:
		for tup in s:
			out.write(', '.join(tup) + '\n\n')

def test(grammar: PCFG = english_grammar_RC_PP, ex_generator: Callable = pres_or_past, *args: Tuple, **kwargs: Dict) -> None:
	"""
	Print out a sample pair of sentences generated by ex_generator(grammar, *args, **kwargs)
	"""
	s, p, t = ex_generator(grammar, *args, **kwargs)
	s = format_tree_string(s)
	t = format_tree_string(t)
	print(f'\n\t{s}\n\t\u27f6 {p}: {t}\n')

# These grammars ARE ONLY USED FOR PARSING during evaluation
# we include the past tense forms as well for parsing the embedded clauses, which are not reinflected
english_grammar_RC_PP_pres_tense = CFG.fromstring("""
	S -> DP VP
	
	DP -> D NP
	NP -> N | NP PP | NP CP
	N -> N_sg | N_pl
	
	VP -> V DP | V EOS
	
	PP -> P DP
	CP -> C VP
	
	D -> 'the'
	
	N_sg -> 'student' | 'professor' | 'headmaster' | 'friend'
	N_sg -> 'assistant'| 'dean'| 'advisor'| 'colleague'
	N_sg -> 'president'| 'chancellor'
	
	N_pl -> 'students'| 'professors'| 'headmasters'| 'friends'
	N_pl -> 'assistants'| 'deans'| 'advisors'| 'colleagues'
	N_pl -> 'presidents'| 'chancellors'
	
	V -> 'help' | 'helps' | 'helped'
	V -> 'visit' | 'visits' | 'visited' 
	V -> 'like' | 'likes' | 'liked'
	V -> 'bother' | 'bothers' | 'bothered'
	V -> 'inspire' | 'inspires' | 'inspired'
	V -> 'recruit' | 'recruits' | 'recruited'
	V -> 'assist' | 'assists' | 'assisted'
	V -> 'confound' | 'confounds' | 'confounded'
	V -> 'accost' | 'accosts' | 'accosted'
	V -> 'avoid' | 'avoids' |  'avoided'
	
	EOS -> '[EOS]'
	
	P -> 'of' | 'near' | 'by' | 'behind' | 'with'
	
	C -> 'that'
""")
setattr(english_grammar_RC_PP_pres_tense, 'lang', 'en_RC_PP')

english_grammar_RC_PP_pres_tense_gen = CFG.fromstring("""
	S -> DP VP
	
	DP -> D NP
	NP -> N | NP PP | NP CP
	N -> N_sg | N_pl
	
	VP -> V DP | V EOS
	
	PP -> P DP
	CP -> C VP
	
	D -> 'the'
	
	N_sg -> 'house' | 'shed' | 'tree' | 'machine'
	N_sg -> 'building' | 'church' | 'office' | 'motel'
	N_sg -> 'barn' | 'shack'
		
	N_pl -> 'houses'| 'sheds'| 'trees'| 'machines'
	N_pl -> 'buildings'| 'churches'| 'offices'| 'motels'
	N_pl -> 'barns'| 'shacks'
	
	V -> 'blocked' | 'blocks' | 'block'
	V -> 'dwarfed' | 'dwarfs' | 'dwarf'
	V -> 'overlooked' | 'overlooks' | 'overlook' 
	V -> 'faced' | 'faces' | 'face' 
	V -> 'predated' | 'predates' | 'predate'
	V -> 'eclipsed' | 'eclipses' | 'eclipse' 
	V -> 'concealed' | 'conceals' | 'conceal' 
	V -> 'shaded' | 'shades' | 'shade'
	V -> 'overshadowed' | 'overshadows' | 'overshadow'
	V -> 'adjoined' | 'adjoins' | 'adjoin'
	
	EOS -> '[EOS]'
	
	P -> 'beside' | 'near' | 'by' | 'behind' | 'around'
	
	C -> 'that'
""")
setattr(english_grammar_RC_PP, 'lang', 'en_RC_PP_gen')