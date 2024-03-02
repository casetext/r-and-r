import os
import json
import pathlib
import time
from copy import deepcopy
from .parsers import chunk_document, parse_page, split_pages
from .prompts import *

def run_page_numbers_only(questions, document, model, repeat_prompt=False, repeat_interval=10000):
	words_in_mouth = '\n{"question": '

	prompt = get_page_numbers_only(questions, document, model.get_tokens, repeat_prompt=repeat_prompt, repeat_interval=repeat_interval)
	response = model.get_response(prompt, return_json=True, words_in_mouth=words_in_mouth)
	if response is None:
		pages = []
	elif type(response) == dict:
		pages = [int(p) for p in response['pages']]
	else:
		pages = [int(p) for l in response for p in l['pages']]

	pages = sorted(list(set(pages)))

	out = {
		'questions': questions, 
		'pages': pages
	}

	return out


def run(model, dataset, question_id=0, total_context=10000, answer_position=0, repeat_prompt=False, repeat_interval=10000, repeat_at_beginning=False, repeat_before_answer=False, repeat_tag_only=False, return_page=False, return_page_only=False, rephrase_count=0, do_abbreviate=False, do_chunking=False, chunk_size=10000, output_path=None, overwrite=False):
	# check args
	if repeat_at_beginning and repeat_before_answer:
		raise ValueError('repeat_at_beginning and repeat_before_answer are incompatible.')
	if do_chunking:
		assert do_abbreviate, 'Chunking requires abbreviation to be on.'
	if repeat_at_beginning or repeat_tag_only:
		assert do_abbreviate == False, 'repeat_at_beginning and repeat_tag_only are not compatible with do_abbreviate.'

	# quit if path exists
	if (not overwrite) and os.path.exists(output_path):
		return None

	# get start time
	start_time = time.time()

	# reset accumulated token count
	model.reset_tokens()

	# load data example
	example = dataset.get(question_id=question_id, answer_position=answer_position, total_context=total_context, get_tokens=model.get_tokens)
	question = example['question']
	document = example['document']

	# get time not including preprocessing
	mid_time = time.time()

	if do_abbreviate:
		questions = [question]

		# get a bunch of variants of the original question
		if rephrase_count > 0:
			words_in_mouth = '\n[\n'
			prompt = query_expansion(question, num=rephrase_count)
			questions_ = model.get_response(prompt, return_json=True, words_in_mouth=words_in_mouth)
			questions.extend(questions_)

			prompt = query_splitter(question, num=rephrase_count)
			questions_ = model.get_response(prompt, return_json=True, words_in_mouth=words_in_mouth)
			questions.extend(questions_)

		# get the most relevant pages from the context using the question variants
		if do_chunking:
			chunks = chunk_document(document, chunk_size, model.get_tokens)
			example['do_abbreviate'] = [run_page_numbers_only(questions, chunk, model, repeat_prompt=repeat_prompt, repeat_interval=repeat_interval) for chunk in chunks]
			pages = sorted(list(set([p for chunk in example['do_abbreviate'] for p in deepcopy(chunk['pages'])])))
		else:
			example['do_abbreviate'] = run_page_numbers_only(questions, document, model, repeat_prompt=repeat_prompt, repeat_interval=repeat_interval)
			pages = deepcopy(example['do_abbreviate']['pages'])

		# create document from extracted pages
		document = '\n\n'.join( filter(lambda p: len(p)>0, [parse_page(document, p) for p in pages]) )

	# put words in Claude's mouth
	question_escaped_quotes = question.replace('"', '\\"')
	words_in_mouth = f'''
{{
    "question": "{question_escaped_quotes}", 
    '''
	words_in_mouth += '"page": ' if return_page_only \
		else '"answer": '

	# for testing reprompt only before answer
	if repeat_before_answer:
		try:
			repeat_before_pages = [example['page']]
		except:
			repeat_before_pages = example['pages']
	else:
		repeat_before_pages = None

	# run the prompt
	prompt = get_answer(question, document, model.get_tokens, repeat_prompt=repeat_prompt, repeat_interval=repeat_interval, repeat_before_pages=repeat_before_pages, repeat_at_beginning=repeat_at_beginning, repeat_tag_only=repeat_tag_only, return_page=return_page, return_page_only=return_page_only)
	example['prompt'] = prompt
	example['response'] = model.get_response(prompt, return_json=True, words_in_mouth=words_in_mouth)

	# record token count
	example['input_tokens'] = model.input_tokens
	example['output_tokens'] = model.output_tokens

	# record time
	end_time = time.time()
	example['model_time'] = end_time - mid_time
	example['total_time'] = end_time - start_time

	# write to disk
	output_dict = {k: v for k, v in example.items() if k not in ['document', 'prompt']}
	if output_path is not None:
		pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, 'w') as f:
			json.dump(output_dict, f, indent=2)
		
	return output_dict
