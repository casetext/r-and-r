import json
import pathlib
import itertools
from anthropic import BadRequestError
from src.datasets import load_dataset
from src.models import load_model
from src.run import run

def saferun(*args, **kwargs):
	try:
		return run(*args, **kwargs)
	except BadRequestError as e:
		if 'Output blocked by content filtering policy' in e.message:
			out = {'response': 'CONTENT BLOCKED'}
			pathlib.Path(kwargs['output_path']).parent.mkdir(parents=True, exist_ok=True)
			with open(kwargs['output_path'], 'w') as f:
				json.dump(out, f, indent=2)
		return None
	except:
		return None


model_name = 'gpt4'
if model_name == 'gpt4':
	model = load_model('openai')
if model_name == 'claude':
	model = load_model('anthropic')

dataset_names = ['nq', 'squad', 'hotpotqa', 'pubmed']
context_len = 80000
answer_positions = range(0, context_len+1, 10000)

for dataset_name in dataset_names:
	dataset = load_dataset(dataset_name)
	question_ids = range(250) if dataset_name == 'hotpotqa' else range(50)
	for (question_id, answer_position) in itertools.product(question_ids, answer_positions):
		if answer_position > 0 and dataset_name == 'hotpotqa':
			continue
		print(f'{dataset_name} {context_len//1000:d}k q{question_id} a{answer_position}')
		for cs in [10, 20, 40, 80]:

			print(f'CR {cs:d}k')
			out = saferun(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, do_abbreviate=True, do_chunking=True, chunk_size=cs*1000, output_path=f'results/cvr/{dataset_name}/{model_name}/c{context_len:d}/cr{cs:d}k/q{question_id:d}/a{answer_position:d}.json')

			if cs == 10:
				continue

			print(f'CR {cs:d}k + Reprompt')
			if cs == 80:
				out = saferun(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, repeat_prompt=True, repeat_interval=10000, do_abbreviate=True, output_path=f'results/cvr/{dataset_name}/{model_name}/c{context_len:d}/cr80k+reprompt/q{question_id:d}/a{answer_position:d}.json')
			else:
				out = saferun(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, repeat_prompt=True, repeat_interval=10000, do_abbreviate=True, do_chunking=True, chunk_size=cs*1000, output_path=f'results/cvr/{dataset_name}/{model_name}/c{context_len:d}/cr{cs:d}k+reprompt/q{question_id:d}/a{answer_position:d}.json')


print('done!')
