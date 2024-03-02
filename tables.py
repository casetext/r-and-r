import os
import json
import re
import itertools
import pathlib
import numpy as np
from src.metrics import fuzzy_match


### Retrieve results

def get_files(root_dir):
	files = []
	for root, dirnames, filenames in os.walk(root_dir):
		for file in filenames:
			if '.json' in file:
				if 'a-1.json' in file:
					continue
				files.append(os.path.join(root, file))

	return files


def get_score(results_dir, dataset, context_len, method, model='gpt4', page=False):
	root_dir = os.path.join(results_dir, dataset, model, f'c{context_len:d}', method)
	files = get_files(root_dir)

	if len(files) == 0:
		return None

	scores = []
	for file in files:
		with open(file, 'r') as f:
			D = json.load(f)
		if D['response'] is None or D['response'] == 'CONTENT BLOCKED':
			score = 0
		elif page:
			score = int(D['response']['page'] == D['page'])
		else:
			score = fuzzy_match(D['response']['answer'], D['answer'])
		scores.append(score)

	score = 100*np.array(scores).mean()
	return score


def get_score_per_answer_position(results_dir, dataset, context_len, method, model='gpt4'):
	root_dir = os.path.join(results_dir, dataset, model, f'c{context_len:d}', method)
	files = get_files(root_dir)

	if len(files) == 0:
		return None

	scores = {}
	for file in files:
		with open(file, 'r') as f:
			D = json.load(f)
		score = 0 if D['response'] is None \
			else fuzzy_match(D['response']['answer'], D['answer'])
		ap = re.search(r'/(a.+?)\.json', file).groups(0)[0]
		if ap not in scores:
			scores[ap] = []
		scores[ap].append(score)

	scores = {a: 100*np.array(s).mean() for a, s in scores.items()}
	return scores


def get_token_usage(results_dir, context_len, method, model='gpt4'):
	datasets = ['nq', 'squad', 'hotpotqa', 'pubmed']
	out = {'input_tokens': [], 'output_tokens': []}
	for dataset in datasets:
		root_dir = os.path.join(results_dir, dataset, model, f'c{context_len:d}', method)
		files = get_files(root_dir)

		if len(files) == 0:
			return None

		for file in files:
			with open(file, 'r') as f:
				D = json.load(f)

			out['input_tokens'].append(D['input_tokens'])
			out['output_tokens'].append(D['output_tokens'])

	for k, v in out.items():
		out[k] = int( np.array(v).mean() )

	return out



### score tables

def tabulate_baseline_vs_reprompt_scores(results_dir, output_path, page=False):
	named_datasets = {'nq': 'NQ', 'squad': r'''\makecell{SQuAD \\ (SQ)}''', 'hotpotqa': r'''\makecell{HotPotQA \\ (HP)}''', 'pubmed': r'''\makecell{PubMed \\ (PM)}'''}
	context_lens = [10000, 20000, 40000, 80000]
	models = ['gpt4', 'claude']
	methods = ['baseline', 'reprompt', 'cr+reprompt']

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{ccS[table-format=3.1]SSSSS}
\toprule
\quad & \quad & \multicolumn{3}{c}{{GPT4}} & \multicolumn{3}{c}{{Claude2}} \\
\cmidrule{3-8}
\quad & $d$ & {Base} & {Rep} & {R\&R} & {Base} & {Rep} & {R\&R} \\
''')

	for dataset, dataset_name in named_datasets.items():
		f.write('\\midrule\n')
		for i, context_len in enumerate(context_lens):
			if i == 0:
				f.write('\\multirow{4}{*}{'+dataset_name+'}')
			else:
				f.write('')
			f.write(f' & {context_len//1000:d}k')
			for (model, method) in itertools.product(models, methods):
				score = get_score(results_dir, dataset, context_len, method, model=model, page=page)
				if score is not None:
					f.write(f' & {score:.1f}')
				else:
					f.write(' &')
			f.write(' \\\\\n')

	f.write('\\bottomrule\n\\end{tabular}''')

	f.close()
	return


def tabulate_cr_wwo_reprompt_scores(results_dir, output_path, page=False):
	named_datasets = {'nq': 'NQ', 'squad': 'SQ', 'hotpotqa': 'HP', 'pubmed': 'PM'}
	chunk_sizes = [10, 20, 40, 80]
	models = ['gpt4', 'claude']
	methods = ['cr{cs:d}k', 'cr{cs:d}k+reprompt']

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{ccSSSS}
\toprule
\quad & \quad & \multicolumn{2}{c}{{GPT4}} & \multicolumn{2}{c}{{Claude2}} \\
\cmidrule{3-6}
\quad & $c$ & {ICR} & {R\&R} & {ICR} & {R\&R} \\
''')

	for dataset, dataset_name in named_datasets.items():
		f.write('\\midrule\n')
		for i, chunk_size in enumerate(chunk_sizes):
			if i == 0:
				f.write('\\multirow{4}{*}{'+dataset_name+'}')
			else:
				f.write('')
			f.write(f' & {chunk_size:d}k')
			for (model, method) in itertools.product(models, methods):
				score = get_score(results_dir, dataset, 80000, method.format(cs=chunk_size), model=model, page=page)
				if score is not None:
					f.write(f' & {score:.1f}')
				else:
					f.write(' &')
			f.write(' \\\\\n')

	f.write('\\bottomrule\n\\end{tabular}''')

	f.close()
	return



### token usage tables

def tabulate_cr_wwo_reprompt_token_usage(results_dir, output_path):
	chunk_sizes = [10, 20, 40, 80]
	llm_calls = [9, 5, 3, 2]

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{ccccc}
\toprule
\quad & \quad & \multicolumn{2}{c}{In} & \quad \\
\cmidrule{3-4}
$c$ & $m$ & ICR & R\&R & Out \\
\midrule
''')

	for chunk_size, llm_call in zip(chunk_sizes, llm_calls):
		f.write(f'{chunk_size:d}k & {llm_call:d}')
		tok_cr = get_token_usage(results_dir, 80000, f'cr{chunk_size:d}k')
		tok_rep = get_token_usage(results_dir, 80000, f'cr{chunk_size:d}k+reprompt')
		in_cr = tok_cr['input_tokens']
		in_rep = tok_rep['input_tokens'] if tok_rep is not None else None
		out = tok_cr['output_tokens']
		if in_rep is None:
			f.write(f' & {in_cr:d} & & {out:d} \\\\\n')
		else:
			f.write(f' & {in_cr:d} & {in_rep:d} & {out:d} \\\\\n')

	f.write(r'''\bottomrule
\end{tabular}''')

	f.close()
	return



### analysis

def tabulate_page_retrieval(results_dir, output_path):
	named_datasets = {'squad': 'SQ', 'pubmed': 'PM'}

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{cSS}
\toprule
\quad & {Answer} & {Page} \\
\midrule
''')

	for dataset, dataset_name in named_datasets.items():
		answer_score = get_score(results_dir, dataset, 40000, 'answer-only', page=False)
		page_score = get_score(results_dir, dataset, 40000, 'page-only', page=True)
		f.write(f'{dataset_name} & {answer_score:2.1f} & {page_score:2.1f} \\\\\n')

	f.write('\\bottomrule\n\\end{tabular}''')

	f.close()
	return


def tabulate_reprompt_mechanism(results_dir, output_path):
	named_datasets = {'nq': 'NQ', 'squad': 'SQ', 'hotpotqa': 'HP', 'pubmed': 'PM'}
	methods = ['reprompt', 'repeat-tag-only', 'repeat-at-beginning', 'repeat-before-answer']

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{cSSSS}
\toprule
\quad & {Rep} & {\makecell{Tags \\ only}} & {\makecell{At \\ beginning \\ only}} & {\makecell{Before \\ answer \\ only}} \\
\midrule
''')

	for dataset, dataset_name in named_datasets.items():
		f.write(f'{dataset_name}')
		for method in methods:
			score = get_score('results/baseline_vs_reprompt', dataset, 40000, 'reprompt') if method=='reprompt' \
				else get_score(results_dir, dataset, 40000, method)
			if score is None:
				f.write(' & \\quad')
			else:
				f.write(f' & {score:.1f}')

		f.write('\\\\\n')

	f.write('\\bottomrule\n\\end{tabular}''')

	f.close()
	return


def tabulate_reprompt_tuning(results_dir, output_path):
	named_datasets = {'nq': 'NQ', 'squad': 'SQ', 'hotpotqa': 'HP', 'pubmed': 'PM'}
	intervals = ['5k', '10k', '20k']

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{cSSS}
\toprule
\quad & {5k} & {10k} & {20k} \\
\midrule
''')

	for dataset, dataset_name in named_datasets.items():
		f.write(f'{dataset_name}')
		for interval in intervals:
			score = get_score(results_dir, dataset, 40000, interval)
			f.write(f' & {score:.1f}')

		f.write('\\\\\n')

	f.write('\\bottomrule\n\\end{tabular}''')

	f.close()
	return



### appendix

def tabulate_baseline_vs_reprompt_scores_per_answer_position(results_dir, dataset, output_path, model='gpt4'):
	context_lens = [10000, 20000, 40000, 80000]
	named_methods = {'baseline': 'Base', 'reprompt': 'Rep', 'cr+reprompt': 'R\&R'}

	pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	f = open(output_path, 'w')

	f.write(r'''\begin{tabular}{ccSSSSSSSSS}
\toprule
\quad & $d$ & {0k} & {10k} & {20k} & {30k} & {40k} & {50k} & {60k} & {70k} & {80k} \\
''')

	for method, method_name in named_methods.items():
		f.write('\\midrule\n')
		for i, context_len in enumerate(context_lens):
			if i == 0:
				f.write('\\multirow{4}{*}{'+method_name+'}')
			else:
				f.write('')
			f.write(f' & {context_len//1000:d}k')
			scores = get_score_per_answer_position(results_dir, dataset, context_len, method, model=model)
			for answer_position in range(0, 80001, 10000):
				ap = f'a{answer_position:d}'
				if scores is not None and ap in scores:
					score = scores[ap]
					f.write(f' & {score:.1f}')
				else:
					f.write(' &')
			f.write(' \\\\\n')

	f.write(r'''\bottomrule
\end{tabular}''''')

	f.close()
	return



if __name__ == '__main__':
	print('Generating main tables . . . ')
	tabulate_baseline_vs_reprompt_scores('results/baseline_vs_reprompt', 'tables/scores/baseline_vs_reprompt.tex')
	tabulate_cr_wwo_reprompt_scores('results/cvr', 'tables/scores/cr_wwo_reprompt.tex')
	tabulate_cr_wwo_reprompt_token_usage('results/cvr', 'tables/cost.tex')

	print('Generating analysis tables . . . ')
	for m in ['page_retrieval', 'reprompt_mechanism', 'reprompt_tuning']:
		vars()[f'tabulate_{m}'](f'results/analysis/{m}', f'tables/analysis/{m}.tex')

	print('Generating appendix tables . . . ')
	for dataset in ['nq', 'squad', 'pubmed']:
		tabulate_baseline_vs_reprompt_scores_per_answer_position('results/baseline_vs_reprompt', dataset, f'tables/triangles/{dataset}.tex')

	print('Done!')
