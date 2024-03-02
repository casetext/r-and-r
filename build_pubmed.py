from src.datasets import PubMed

pm = PubMed()

print('Collecting abstracts . . . ')
pm.collect_abstracts()

print('Generating questions . . . ')
pm.generate_questions(num_questions=250)

print('Done!')
