import os
import shutil
import logging
import traceback

logger = logging.getLogger(__name__)

class SafeSharedTemporaryDirectory():
	'''Creates and closes a tmp dir that may be shared across processes.'''
	def __init__(self, dir: str = 'tmp'):
		self.dir = dir
	
	def __enter__(self):
		os.makedirs(self.dir, exist_ok=True)
		return self.dir
	
	def __exit__(self, exc_type, exc_value, tb):
		if exc_type is not None:
			traceback.print_exception(exc_type, exc_value, tb)
		
		try:
			shutil.rmtree(self.dir, ignore_errors=True)
		except PermissionError as e:
			logger.warning(
				f'The directory {self.dir} will not be '
				'deleted because it is being used by another process.'
			)
			pass