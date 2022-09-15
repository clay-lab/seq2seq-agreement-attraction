# code adapted from Bob Frank's grammars.py
import random

from nltk import Tree, PCFG
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
	'sg': KeyBasedDefaultDict(lambda s: s.replace('ed', 's')),
	'pl': KeyBasedDefaultDict(lambda s: s.replace('ed',  ''))
}

PAST_PRES['sg'].update({
	k: k.replace('ed', 'es') for k in [
		'liked', 
		'inspired',
	]
})

PAST_PRES['pl'].update({
	k: k.replace('ed', 'e') for k in [
		'liked', 
		'inspired',
	]
})

english_grammar = PCFG.fromstring("""
	S -> DP VP [1.0]
	
	DP -> D NP [1.0]
	NP -> N [0.8] | NP PP [0.1] | NP CP [0.1]
	N -> N_sg [0.5] | N_pl [0.5]
	
	VP -> V DP [1.0]
	
	PP -> P DP [1.0]
	CP -> C VP [1.0]
	
	D -> 'the' [1.0]
	
	N_sg -> 'student' [0.25]  | 'professor' [0.25]  | 'headmaster' [0.25]  | 'friend' [0.25]
	N_pl -> 'students' [0.25] | 'professors' [0.25] | 'headmasters' [0.25] | 'friends' [0.25] 
	
	V -> 'helped' [0.1] | 'visited' [0.1] | 'liked' [0.1] 
	V -> 'bothered' [0.1] | 'inspired' [0.1] | 'recruited' [0.1] 
	V -> 'assisted' [0.1] | 'confounded' [0.1] | 'accosted' [0.1] | 'avoided' [0.1]
	
	P -> 'of' [0.2] | 'near' [0.2] | 'by' [0.2] | 'behind' [0.2] | 'with' [0.2]
	
	C -> 'that' [1.0]
""")
setattr(english_grammar, 'lang', 'en')

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
	main_clause_VP = grep_next_subtree(t_copy, 'VP')
	main_clause_V = grep_next_subtree(main_clause_VP, r'^V$')
	
	# get the number of the main clause subject
	main_clause_subject = grep_next_subtree(t_copy, r'^N_')
	subject_number = 'sg' if main_clause_subject.label().symbol().endswith('sg') else 'pl'
	
	# map the past form of the verb to the present form based on the number of the subject
	main_clause_V[0] = PAST_PRES[subject_number][main_clause_V[0]]
	
	return t_copy

def pres_or_past(grammar: PCFG, pres_p: float = 0.5) -> Tuple:
	
	return present_pair(grammar) if random.random() < pres_p else past_pair(grammar)

def pres_or_past_no_pres_dist(grammar: PCFG, pres_p: float = 0.5) -> Tuple:
	
	source, pfx, target = present_pair(grammar) if random.random() < pres_p else past_pair(grammar)
	
	# for English, we do not care about distractors in the past tense, since they do not affect attraction
	# in fact, we WANT some of these for training
	if pfx == 'pres':
		# otherwise, we need to modify the tree to remove the distractor(s) from the source and the target
		subject_position = [
			position 
			for position in source.treepositions() 
				if not isinstance(source[position],str) and str(source[position].label()) == 'NP'
		][0]
		
		for tree in [source, target]:
			main_clause_subject = grep_next_subtree(tree, r'^NP$').copy(deep=True)
			main_clause_subject_no_dist = grep_next_subtree(main_clause_subject, r'^N$')
			tree[subject_position] = main_clause_subject_no_dist
		
	return source, pfx, target

def test_file(grammar: PCFG = english_grammar, n: int = 10, filename: str = 'test.txt'):
	"""
	Create a small test file with n pairs of formatted present and past tense sentences
	"""
	s = [present_pair(grammar) for _ in range(n)]
	s = [(format_tree_string(past_tree), pfx, format_tree_string(present_tree)) for past_tree, pfx, present_tree in s]
	with open(filename, 'w') as out:
		for tup in s:
			out.write(', '.join(tup) + '\n\n')

# This grammar IS ONLY USED FOR PARSING during evaluation
# we include the past tense forms as well for parsing the embedded clauses, which are not reinflected
english_grammar_pres_tense = PCFG.fromstring("""
	S -> DP VP [1.0]
	
	DP -> D NP [1.0]
	NP -> N [0.8] | NP PP [0.1] | NP CP [0.1]
	N -> N_sg [0.5] | N_pl [0.5]
	
	VP -> V DP [1.0]
	
	PP -> P DP [1.0]
	CP -> C VP [1.0]
	
	D -> 'the' [1.0]
	
	N_sg -> 'student' [0.25]  | 'professor' [0.25]  | 'headmaster' [0.25]  | 'friend' [0.25]
	N_pl -> 'students' [0.25] | 'professors' [0.25] | 'headmasters' [0.25] | 'friends' [0.25] 
	
	V -> 'help' [0.033] | 'helps' [0.033] | 'helped' [0.034]
	V -> 'visit' [0.033] | 'visits' [0.033] | 'visited' [0.034] 
	V -> 'like' [0.033] | 'likes' [0.033] | 'liked' [0.034]
	V -> 'bother' [0.033] | 'bothers' [0.033] | 'bothered' [0.034]
	V -> 'inspire' [0.033] | 'inspires' [0.033] | 'inspired' [0.034]
	V -> 'recruit' [0.033] | 'recruits' [0.033] | 'recruited' [0.034]
	V -> 'assist' [0.033] | 'assists' [0.033] | 'assisted' [0.034]
	V -> 'confound' [0.033] | 'confounds' [0.033] | 'confounded' [0.034]
	V -> 'accost' [0.033] | 'accosts' [0.033] | 'accosted' [0.034]
	V -> 'avoid' [0.033] | 'avoids' [0.033] |  'avoided' [0.034]
	
	P -> 'of' [0.2] | 'near' [0.2] | 'by' [0.2] | 'behind' [0.2] | 'with' [0.2]
	
	C -> 'that' [1.0]
""")
setattr(english_grammar_pres_tense, 'lang', 'en')
