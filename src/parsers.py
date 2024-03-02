import re
import ast
import json
import numpy as np
from copy import deepcopy

### find a JSON inside a string

def parse_single_json(text):
	opening_pos = None
	count = 0

	for pos, char in enumerate(text):

		if opening_pos is not None and char in [opening_char, closing_char]:
			count += (1 if char == opening_char else -1)
			if count == 0:
				closing_pos = pos
				D = text[opening_pos:closing_pos+1]
				next_text = text[closing_pos+1:]
				return D, next_text

		if opening_pos is None and char in ['[', '{']:
			opening_pos = pos
			opening_char = char
			closing_char = ']' if char == '[' else '}'
			count += 1

	return None, None


def parse_json(text):
	Ds = []

	while True:
		D, text = parse_single_json(text)
		if D is None:
			break
		try:
			Ds.append(ast.literal_eval(D))
		except:
			if D.startswith('[') and D.endswith(']'):
				D = ["'" + l.strip().strip(',')[1:-1] + "'" for l in D[1:-1].split('\n') if l.strip() != '']
				Ds.append(D)
			elif D.startswith('{"question":  "') and '", "pages": ' in D:
				i = len('{"question":  "')
				j = D.find('", "pages": ')
				D = D[:i] + D[i:j].replace('"', '\\"') + D[j:]
				Ds.append(ast.literal_eval(D))
			elif D[D.find('"page": ')+8].isalpha():
				D = D.replace('"page": ', '"page": "').replace('\n}', '"\n}')
				Ds.append(ast.literal_eval(D))
			else:
				raise SyntaxError('The following string cannot be parsed as JSON:\n\n' + D)

	if len(Ds) == 0:
		return None

	if len(Ds) == 1:
		return Ds[0]

	return Ds


### parse page XML block from document

def parse_page(document, page):
	header = f'<PAGE {page}>'
	footer = f'</PAGE {page}>'

	if header not in document:
		return ''

	text = document.split(header)[1].split(footer)[0]
	return header + text + footer


def split_pages(document):
	chunks = document.strip().split('\n\n<PAGE')
	chunks = [chunks[0]] + ['<PAGE'+chunk for chunk in chunks[1:]]
	return chunks


### chunking and maximizing uniformity

def balance_chunks(A, B):
	d = sum(A) - sum(B)

	if d == 0:
		return A, B

	if d < 0:
		B, A = balance_chunks(list(reversed(B)), list(reversed(A)))
		return list(reversed(A)), list(reversed(B))

	while abs(d - 2*A[-1]) < abs(d):
		d -= 2*A[-1]
		B.insert(0, A.pop())

	return A, B


def max_uniform_partition(data, chunk_size):
	chunks = []

	chunk = []
	count = 0
	for x in data:
		chunk.append(x)
		count += x
		if count >= chunk_size:
			chunks.append(chunk)
			chunk = []
			count = 0

	if len(chunk) > 0:
		chunks.append(chunk)

	while True:
		old_chunks = deepcopy(chunks)
		for i in range(len(chunks)-2, -1, -1):
			chunks[i], chunks[i+1] = balance_chunks(chunks[i], chunks[i+1])
		if old_chunks == chunks:
			break

	return chunks


def chunk_document(document, chunk_size, token_counter):
	pages = split_pages(document)
	pages = [p+'\n\n' for p in pages]
	counts = [token_counter(p) for p in pages]

	chunked_counts = max_uniform_partition(counts, chunk_size)
	page_counts = np.array([len(chunk) for chunk in chunked_counts])
	idxs = np.insert(np.cumsum(page_counts), 0, 0)
	chunks = [''.join(pages[i:j]).strip() for i, j in zip(idxs[:-1], idxs[1:])]

	return chunks
