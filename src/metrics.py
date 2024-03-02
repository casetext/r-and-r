import re
import string


def fuzzy_match(a, b):
	assert type(a) == str, 'The first argument of fuzzy_match must be a string.'
	if type(b) == list:
		return max(fuzzy_match(a, m) for m in b)
	else:
		assert type(b) == str, 'The second argument of fuzzy_match must either be a string or nested list of strings.'

	a = a.lower().strip()
	b = b.lower().strip()

	if a != 'n/a':
		a = re.sub(r'[^a-z0-9 ]+', '', a)
		a = re.sub(' +', ' ', a)

	if b != 'n/a':
		b = re.sub(r'[^a-z0-9 ]+', '', b)
		b = re.sub(' +', ' ', b)

	a_set = set(a.split())
	b_set = set(b.split())

	return int( a_set.issubset(b_set) or b_set.issubset(a_set) or a in b or b in a )
