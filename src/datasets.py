import os
import json
import functools
from xopen import xopen
from .models import load_model



def paginate(func):

	def wrapper(*args, **kwargs):
		get_tokens = kwargs['get_tokens']
		question, answer, context, filler = func(*args, **kwargs)

		context = {'text': '\n<PAGE {PAGE}>\n' + context + '\n</PAGE {PAGE}>\n'}
		context['tokens'] = get_tokens(context['text'])

		filler = [{'text': '\n<PAGE {PAGE}>\n' + f + '\n</PAGE {PAGE}>\n'} for f in filler]
		count = 0
		for f in filler:
			f['tokens'] = get_tokens(f['text'])
			count += f['tokens']
			if count >= 100000:
				break

		filler = [f for f in filler if 'tokens' in f.keys()]

		return question, answer, context, filler

	return wrapper



class LocalContextDataset(object):

	def __init__(self):
		return


	@functools.lru_cache(maxsize=1)
	@paginate
	def get_materials(self, question_id=0, get_tokens=None):
		raise NotImplemented


	@functools.lru_cache(maxsize=1)
	def get(self, question_id=0, answer_position=0, total_context=100000, get_tokens=None):
		# get raw materials
		question, answer, context, filler = self.get_materials(question_id=question_id, get_tokens=get_tokens)

		page_counter = 1
		total_len = 0
		added = answer_position<0

		# start creating document
		document = ''
		for junk in filler:

			document += junk['text'].replace('{PAGE}', f'{page_counter:d}')
			page_counter += 1
			total_len += junk['tokens']
		
			# add the actual context containing the answer at `answer_position`
			if total_len >= answer_position and not added:
				added = True
				document += context['text'].replace('{PAGE}', f'{page_counter:d}')
				page_num = page_counter
				page_counter += 1
				total_len += context['tokens']
		
			# don't exceed the `total_context` defined
			if total_len >= total_context:
				break

		# if testing for hallucinations
		if answer_position < 0:
			answer = 'n/a'
			page_num = 'n/a'

		out = {
			'question': question, 
			'answer': answer, 
			'page': page_num, 
			'document': document.strip()
		}

		return out



class NQ(LocalContextDataset):

	def __init__(self, path='data/nq/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz'):
		super(NQ, self).__init__()
		self.path = path

		#load NQ dataset
		with xopen(self.path, 'r') as f:
			self.dataset = [json.loads(l) for i, l in enumerate(f) if (i < 255 and i not in [13, 32, 40, 164, 216])]


	@functools.lru_cache(maxsize=1)
	@paginate
	def get_materials(self, question_id=0, get_tokens=None):
		example = self.dataset[question_id]

		question = example['question']
		answer = example['answers']
		context = example['nq_annotated_gold']['chunked_long_answer'].strip()

		filler = [doc['text'] for doc in example['ctxs'] if not doc['hasanswer']]

		return question, answer, context, filler



class Squad(LocalContextDataset):

	def __init__(self, path='data/squad/train-v2.0.json'):
		super(Squad, self).__init__()
		self.path = path

		#load squad dataset
		with open(self.path, 'r') as f:
			self.dataset = json.load(f)


	@functools.lru_cache(maxsize=1)
	@paginate
	def get_materials(self, question_id=0, get_tokens=None):
		example = self.dataset['data'][question_id]['paragraphs'][0]

		question = example['qas'][0]['question']
		answer = example['qas'][0]['answers'][0]['text'].lower()
		context = example['context'].strip()

		filler = [par['context'] for i, topic in enumerate(self.dataset['data']) for par in topic['paragraphs'] if i != question_id]

		return question, answer, context, filler


	
class HotPotQA(object):

	def __init__(self, path='data/hotpotqa/hotpot_train_v1.1.json'):
		self.path = path

		#load hotpotqa dataset
		with open(self.path, 'r') as f:
			hotpot = json.load(f)
		self.dataset = [topic for topic in hotpot if topic['level']=='hard']


	@functools.lru_cache(maxsize=1)
	def get(self, question_id=0, answer_position=0, total_context=100000, get_tokens=None):
		# note: answer_position is only used to specify if we're testing for hallucinations

		# ground truth answer and paragraphs
		target = self.dataset[question_id]
		question = target['question']
		answer = target['answer']
		contexts = [' '.join(c[1]) for c in target['context']]
	
		# everything else is junk filler
		noise = []
		for i,topic in enumerate(self.dataset):
			if i != question_id:
				noise_ = [' '.join(c[1]) for c in topic['context']]
				noise.extend(noise_)
			if len(noise) >= 5000:
				break

		# start creating the context/document
		document = ''
		page_nums = []
		page_counter = 1
		total_len = 0
	
		# since there are multiple relevant paragraphs, split them evenly across context
		l = len(contexts)
		interval = int(total_context/l)
		for j,context in enumerate(contexts):
		
			# use artificial page numbers
			if answer_position >= 0:  # negative means testing hallucinations
				text = f'\n<PAGE {page_counter}>\n{context}\n</PAGE {page_counter}>\n'
				document += text
				page_nums.append(page_counter)
				page_counter += 1
				total_len += get_tokens(text)

			# add filler garbage until we reach the next relevant paragraph
			while total_len < (j+1)*interval:
				garbage = noise.pop()
				text = f'\n<PAGE {page_counter}>\n{garbage}\n</PAGE {page_counter}>\n'
				document += text
				page_counter += 1
				total_len += get_tokens(text)

		# if testing for hallucinations
		if answer_position < 0:
			answer = 'n/a'

		out = {
			'question': question, 
			'answer': answer, 
			'pages': page_nums, 
			'document': document.strip()
		}

		return out



class PubMed(LocalContextDataset):

	def __init__(self, input_file='../pubmed/processed/abstracts/2024.json', output_dir='data/pubmed'):
		super(PubMed, self).__init__()
		self.input_file = input_file
		self.output_dir = output_dir
		self.dataset = None

		os.makedirs(self.output_dir, exist_ok=True)


	def collect_abstracts(self, min_tokens=150, max_tokens=200):
		get_tokens = load_model('openai').get_tokens

		with open(self.input_file, 'r') as f:
			D = f.readlines()

		D = [l.split('\t', 1)[1].strip() for l in D if '\t' in l]
		for i, abstract in enumerate(D):
			if abstract.startswith('"') and abstract.endswith('"'):
				D[i] = abstract[1:-1].strip()

		D = [abstract for abstract in D if min_tokens <= get_tokens(abstract) <= max_tokens]
		D.reverse()

		with open(os.path.join(self.output_dir, 'abstracts.json'), 'w') as f:
			json.dump(D, f, indent=2)

		return


	def generate_questions(self, num_questions=10):
		model = load_odel('openai')

		with open(os.path.join(self.output_dir, 'abstracts.json'), 'r') as f:
			abstracts = json.load(f)

		abstracts = abstracts[:num_questions]

		prompt_template = '''<INSTRUCTIONS>
Write a question that can only be answered by reading the document provided below and not based on any extraneous information. The answer should be a single word or short phrase. Don't refer to the document directly, such as "in the document below"; i.e., the question should make sense even if the document is placed in a collection of documents.

Return your response in JSON format with the following keys:
`question` (str): the generated question
`answer` (list[str]): up to 5 variations of the answer to the question
</INSTRUCTIONS>

<DOCUMENT>
{document}
</DOCUMENT>

<INSTRUCTIONS>
Now, write a question that can only be answered by reading the document above and not based on any extraneous information. The answer should be a single word or short phrase. Don't refer to the document directly, such as "in the document below"; i.e., the question should make sense even if the document is placed in a collection of documents.

Return your response in JSON format with the following keys:
`question` (str): the generated question
`answer` (list[str]): up to 5 variations of the answer to the question
</INSTRUCTIONS>'''

		out = []
		for i, abstract in enumerate(abstracts):
			print(f'question {i:d} . . . ')
			prompt = prompt_template.format(document=abstract)
			qa = model.get_response(prompt, return_json=True)
			qa['context'] = abstract
			out.append(qa)

		with open(os.path.join(self.output_dir, 'questions.json'), 'w') as f:
			json.dump(out, f, indent=2)

		return


	@functools.lru_cache(maxsize=1)
	@paginate
	def get_materials(self, question_id=0, get_tokens=None):
		if self.dataset is None:
			self.dataset = {}
			for thing in ['abstracts', 'questions']:
				with open(os.path.join(self.output_dir, f'{thing}.json'), 'r') as f:
					self.dataset[thing] = json.load(f)

		qa = self.dataset['questions'][question_id]
		question = qa['question']
		answer = qa['answer']
		context = qa['context']

		filler = self.dataset['abstracts'][:]
		filler.pop(question_id)

		return question, answer, context, filler



def load_dataset(dataset_name, **kwargs):
	dataset_dict = {'nq': NQ, 'squad': Squad, 'hotpotqa': HotPotQA, 'pubmed': PubMed}
	dataset = dataset_dict[dataset_name.lower()](**kwargs)
	return dataset
