import pandas as pd

file_path = 'dataset/DBLPOnlyCitationOct19.txt'

data = []

with open(file_path, 'r') as file:
    current_paper = {}
    for line in file:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.startswith('#*'):
            if current_paper:
                current_paper['References'] = ', '.join(current_paper.get('References', [])) if current_paper.get(
                    'References') else None
                data.append(current_paper)
            current_paper = {'Title': line[2:].strip(), 'References': []}
        elif line.startswith('#@'):
            current_paper['Authors'] = line[2:].strip()
        elif line.startswith('#t'):
            current_paper['Year'] = line[2:].strip()
        elif line.startswith('#c'):
            current_paper['Venue'] = line[2:].strip()
        elif line.startswith('#index'):
            current_paper['Index'] = line[6:].strip()
        elif line.startswith('#%'):
            current_paper['References'].append(line[2:].strip())
        elif line.startswith('#!'):
            current_paper['Abstract'] = line[2:].strip()
    if current_paper:
        current_paper['References'] = ', '.join(current_paper.get('References', [])) if current_paper.get(
            'References') else None
        data.append(current_paper)

df = pd.DataFrame(data)

default_columns = ['Title', 'Authors', 'Year', 'Venue', 'Index', 'References', 'Abstract']
for col in default_columns:
    if col not in df.columns:
        df[col] = None

df = df[['Title', 'Authors', 'Year', 'Venue', 'Index', 'References', 'Abstract']]

df.to_csv('dataset/citation-cooked.csv', index=False)
