import itertools
from src.datasets import load_dataset
from src.models import load_model
from src.run import run

model = load_model('openai')

dataset_names = ['nq', 'squad', 'hotpotqa', 'pubmed']
context_lens = [40000]

for dataset_name in dataset_names:
	dataset = load_dataset(dataset_name)
	question_ids = range(250) if dataset_name == 'hotpotqa' else range(50)
	for context_len in context_lens:
		answer_positions = range(0, context_len+1, 10000)
		for (question_id, answer_position) in itertools.product(question_ids, answer_positions):
			if answer_position > 0 and dataset_name == 'hotpotqa':
				continue
			print(f'{dataset_name} {context_len//1000:d}k q{question_id} a{answer_position}')

			print('reprompt 5k')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, repeat_prompt=True, repeat_interval=5000, return_page=True, output_path=f'results/analysis/reprompt_tuning/{dataset_name}/gpt4/c{context_len:d}/5k/q{question_id:d}/a{answer_position:d}.json')

			print('reprompt 20k')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, repeat_prompt=True, repeat_interval=20000, return_page=True, output_path=f'results/analysis/reprompt_tuning/{dataset_name}/gpt4/c{context_len:d}/20k/q{question_id:d}/a{answer_position:d}.json')


			print('repeat before answer')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, repeat_prompt=True, repeat_interval=10000, repeat_before_answer=True, output_path=f'results/analysis/reprompt_mechanism/{dataset_name}/gpt4/c{context_len:d}/repeat-before-answer/q{question_id:d}/a{answer_position:d}.json')

			print('repeat at beginning')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, repeat_prompt=True, repeat_interval=10000, repeat_at_beginning=True, output_path=f'results/analysis/reprompt_mechanism/{dataset_name}/gpt4/c{context_len:d}/repeat-at-beginning/q{question_id:d}/a{answer_position:d}.json')

			print('repeat tag only')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page=True, repeat_prompt=True, repeat_interval=10000, repeat_tag_only=True, output_path=f'results/analysis/reprompt_mechanism/{dataset_name}/gpt4/c{context_len:d}/repeat-tag-only/q{question_id:d}/a{answer_position:d}.json')


			if dataset_name in ['nq', 'hotpotqa']:
				continue

			print('answer only')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, output_path=f'results/analysis/page_retrieval/{dataset_name}/gpt4/c{context_len:d}/answer-only/q{question_id:d}/a{answer_position:d}.json')

			print('page only')
			out = run(model, dataset, question_id=question_id, answer_position=answer_position, total_context=context_len, return_page_only=True, output_path=f'results/analysis/page_retrieval/{dataset_name}/gpt4/c{context_len:d}/page-only/q{question_id:d}/a{answer_position:d}.json')


print('done!')
