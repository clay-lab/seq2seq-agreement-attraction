from typing import *

MUELLER_T5_MODELS: Set[str] = set(
	[f'{pfx}-1m' for pfx in ['babyt5', 'c4', 'wikit5', 'simplewiki']] +
	['babyt5-5m'] +
	[m for pfx in ['c4', 'wikit5', 'simplewiki']
		for m in 
		[f'{pfx}-{i}m' for i in [10, 100]]
	] +
	[m for pfx in ['c4', 'wikit5']
		for m in
		[f'{pfx}-{i}' for i in ['100m_withchildes', '1b']]
	]
)

ALL_MODELS: Set[str] = set(
	[f'google/t5-{size}' 
		for size in [
			'efficient-tiny', 
			'efficient-mini', 
			'efficient-small', 
			'efficient-base',
		]
		# + ['efficient-large', 'efficient-xl', 'efficient-xxl']
	] + 
	[f'google/t5-efficient-base-{ablation}'
		for ablation in 
		[f'dl{i}' for i in range(2,9,2)] +
		[f'el{i}' for i in range(2,9,2)] +
		[f'nl{i}' for i in (2**i for i in range(1,4,1))]
	] +
	[f'google/t5-efficient-mini-{ablation}'
		for ablation in 
		[f'nl{i}' for i in [6, 8, 12, 24]]
	]
)

ALL_MODELS: Set[str] = ALL_MODELS.union(MUELLER_T5_MODELS)

DEFAULT_N_EPOCHS: int = 30