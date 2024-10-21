from colorama import Fore, Style
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-A', action='store_true')
args = parser.parse_args()


'''
\begin{table}[h]
\centering
\begin{tabular}{r|l|l}
\textbf{Identifier}   & \textbf{Activity} & \textbf{Timestamp} \\ \hline
11      & Order product   & 2020-09-24 14:05 \\ 
11      & Payment         & 2020-09-24 14:10 \\ 
12      & Order product   & 2020-09-24 16:00 \\ 
11      & Prepare product & 2020-09-24 16:05 \\ 
12      & Prepare product & 2020-09-24 16:05 \\ 
11      & Send product    & 2020-09-24 16:45 \\ 
12      & Payment         & 2020-09-25 08:30 \\ 
12      & Send product    & 2020-09-25 10:30 \\ 
11      & Receive product & 2020-09-27 13:00 \\ 
$\dots$ & $\dots$         & $\dots$          \\ 
\end{tabular}
\caption{Example event log}\label{tab:event_log}
\end{table}
'''



def make_latex(table, datasetname):
	caption = f'Conformance metrics for dataset {" ".join(datasetname.split("_"))}.'
	label = f'tab:conformance_metrics_{datasetname}'
	latex_table = r'''
\begin{table}[h]
\centering
\begin{tabular}{l|r|r|r|r|r}
\textbf{Method}   & \textbf{fScore} & \textbf{Fitness} & \textbf{Precision} & \textbf{Generalization} & \textbf{Simplicity} \\ \hline
'''
	for index, row in enumerate(table[3:-1]):
		splitted = [''.join(l.split(' ')) for l in row.split('|')[1:-3]]
		splitted = splitted[:5] + [splitted[-1]]
		latex_table += f'{splitted[0].split("_")[0].capitalize()} Miner & {" & ".join(splitted[1:])} '
		if index != len(table[3:-1]) - 1:
			latex_table += '\\\\ \\hline \n'
	latex_table += r'''
\end{tabular}
\caption{''' + caption + r''' }\label{''' + label + r'''}
\end{table}
	'''
	print(latex_table)
	with open(f'/home/dominique/TUe/thesis/report/tables/conformance_{datasetname}.tex', 'w') as file:
		file.write(latex_table)

basedir = '/home/dominique/TUe/thesis/git_data/evaluation_data/'
for directory in sorted(os.listdir(basedir)):
	directory = basedir + directory
	if os.path.isdir(directory):
		print(f'\n{directory}')
		try:
			with open(f'{directory}/predictions/data_conformance{"" if args.A else "_best"}.txt', 'r') as file:
				lines = list(file.readlines())
				# make_latex(lines, directory)
				for line in lines:
					line = line[:-1]
					if 'gcn' in line:
						print(f'{Fore.CYAN}{line}{Style.RESET_ALL}')
					else:
						print(line)
		except FileNotFoundError:
			print('file not found')
