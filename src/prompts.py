import re
from .parsers import split_pages

### Simple Prompting/Reprompting

def get_single_page(question, document):	
	return f'''<INSTRUCTIONS>
Below is a document that is separated into page numbers. Identify the page number in the document that is most relevant to answering the following question: 
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being asked
`page` (int): the most relevant page number to the question, or 'n/a' if no page is relevant.
</INSTRUCTIONS>	

<DOCUMENT>
{document}
</DOCUMENT>

<INSTRUCTIONS>
Now, identify the page number in the document that is most relevant to answering the following question: 
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being asked
`page` (int): the most relevant page number to the question, or 'n/a' if no page is relevant.
</INSTRUCTIONS>'''


def get_answer(question, document, get_tokens, repeat_prompt=False, repeat_interval=10000, repeat_before_pages=None, repeat_at_beginning=False, repeat_tag_only=False, return_page=False, return_page_only=False):
	if repeat_before_pages is not None:
		repeat_prompt = True

	if return_page_only:
		return get_single_page(question, document)

	# if return_page is on, include additional JSON key in prompt
	return_page_prompt = f'''
`page` (int): the page number of the document that contains the answer, or 'n/a' if the answer does not appear in the document''' if return_page \
	else ''

	# if repeat_prompt is on, repeat prompt every `interval` tokens
	if repeat_prompt:
		reprompt = f'''<INSTRUCTIONS_REMINDER>
Remember, your task is to follow the instructions under the '<INSTRUCTIONS>' tag.
</INSTRUCTIONS_REMINDER>

''' if repeat_tag_only \
			else f'''<INSTRUCTIONS_REMINDER>
Remember, your task is to answer the following question based on this document and no additional extraneous information:
{question}

Return your answer in JSON format with the following keys:
`question` (str): the question being answered
`answer` (str): the answer to the question, or 'n/a' if the answer does not appear in the document{return_page_prompt}
</INSTRUCTIONS_REMINDER>

'''
		total_l = 0
		document_ = ''
		chunks = split_pages(document)
		for chunk in chunks:
			if repeat_before_pages is not None:
				page_num = int( re.search(r'<PAGE (\d+)>', chunk).groups(0)[0] )
				if page_num in repeat_before_pages:
					document_ += reprompt
				document_ += chunk + '\n\n'
				continue
			document_ += chunk + '\n\n'
			l = get_tokens(chunk)
			total_l += l
			if total_l >= repeat_interval:
				total_l = 0
				if repeat_at_beginning:
					document_ = reprompt + document_
				else:
					document_ += reprompt

		document = document_
	
	return f'''<INSTRUCTIONS>
Answer the following question based on the document provided and no additional extraneous information:
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being answered
`answer` (str): the answer to the question, or 'n/a' if the answer does not appear in the document{return_page_prompt}
</INSTRUCTIONS>	

<DOCUMENT>
{document}
</DOCUMENT>

<INSTRUCTIONS>
Now, answer the following question based on the above document and no additional extraneous information:
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being answered
`answer` (str): the answer to the question, or 'n/a' if the answer does not appear in the document{return_page_prompt}
</INSTRUCTIONS>'''


### Rephrasing + Reprompting + Abbreviation

def query_expansion(question, num=5):
	return f'''<INSTRUCTIONS>
Consider the following question:
{question}

Generate {num:d} variants of this question that ask the same thing. Return your answer as a Python list containing the {num:d} variants.
</INSTRUCTIONS>'''


def query_splitter(question, num=5):
	return f'''<INSTRUCTIONS>
Consider the following question:
{question}

Break down this question into {num:d} subquestions. Return your answer as a Python list containing the {num:d} subquestions.
</INSTRUCTIONS>'''


def get_page_numbers_only_single_question(question, document, get_tokens, repeat_prompt=False, repeat_interval=10000, num_pages=5):
		
	if repeat_prompt:
		total_l = 0
		document_ = ''
		chunks = split_pages(document)
		for chunk in chunks:
			document_ += chunk + '\n\n'
			l = get_tokens(chunk)
			total_l += l
			if total_l >= repeat_interval:
				total_l = 0
				document_ += f'''<INSTRUCTIONS_REMINDER>
Remember, your task is to identify up to {num_pages:d} page numbers in the document that are most relevant to the following question: 
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
</INSTRUCTIONS_REMINDER>

'''
		document = document_
	
	return f'''<INSTRUCTIONS>
Below is a document that is separated into page numbers. Identify up to {num_pages:d} page numbers in the document that are most relevant to the following question: 
{question}
	
Return your answer in JSON format with the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
</INSTRUCTIONS>
	
<DOCUMENT>
{document}
</DOCUMENT>

<INSTRUCTIONS>
Now, identify up to {num_pages:d} page numbers in the document that are most relevant to the following question. 
{question}

Return your answer in JSON format with the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
</INSTRUCTIONS>'''


def get_page_numbers_only(questions, document, get_tokens, repeat_prompt=False, repeat_interval=10000, num_pages=5):
	if len(questions) == 1:
		question = questions[0]
		return get_page_numbers_only_single_question(question, document, get_tokens, repeat_prompt=repeat_prompt, repeat_interval=repeat_interval, num_pages=num_pages)
	
	questions = '\n'.join(questions)
	
	if repeat_prompt:
		total_l = 0
		document_ = ''
		chunks = split_pages(document)
		for chunk in chunks:
			document_ += chunk + '\n\n'
			l = get_tokens(chunk)
			total_l += l
			if total_l >= repeat_interval:
				total_l = 0
				document_ += f'''<INSTRUCTIONS_REMINDER>
Remember, for each of the following questions, your task is to identify up to {num_pages:d} page numbers in the document that are most relevant to that question: 
{questions}

Return your answers in JSONL format with one question per line. Each line should have the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
Make sure to include every question in the JSONL.
</INSTRUCTIONS_REMINDER>

'''
		document = document_
	
	return f'''<INSTRUCTIONS>
Below is a document that is separated into page numbers. For each of the following questions, identify up to {num_pages:d} page numbers in the document that are most relevant to that question: 
{questions}
	
Return your answers in JSONL format with one question per line. Each line should have the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
Make sure to include every question in the JSONL.
</INSTRUCTIONS>
	
<DOCUMENT>
{document}
</DOCUMENT>

<INSTRUCTIONS>
Now, for each of the following questions, identify up to {num_pages:d} page numbers in the document that are most relevant to that question. 
{questions}

Return your answers in JSONL format with one question per line. Each line should have the following keys: 
`question` (str): the question being answered
`pages` (list[int]): up to {num_pages:d} page numbers of the document that are most relevant to the question.
Make sure to include every question in the JSONL.
</INSTRUCTIONS>'''
