import re

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