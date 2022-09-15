import collections

# from https://stackoverflow.com/questions/7963755/using-the-key-in-collections-defaultdict, user2124834
class KeyBasedDefaultDict(collections.defaultdict):
	def __missing__(self, key):
		if self.default_factory is None:
			raise KeyError(key)
		self[key] = self.default_factory(key)
		return self[key]