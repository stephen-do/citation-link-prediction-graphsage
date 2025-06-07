from typing import List
import pandas as pd


def parse_dblp_citation_file(file_path: str) -> List[dict]:
    """
    Parse the DBLP citation text file into a structured list of dictionaries.

    Args:
        file_path (str): Path to the raw DBLP citation file.

    Returns:
        List[dict]: A list of paper metadata, each as a dictionary.
    """
    data = []
    current_paper = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Start of a new paper entry
            if line.startswith('#*'):
                if current_paper:
                    # Join reference list into a single string
                    current_paper['References'] = ', '.join(current_paper.get('References', [])) \
                        if current_paper.get('References') else None
                    data.append(current_paper)

                # Initialize new paper
                current_paper = {
                    'Title': line[2:].strip(),
                    'References': []
                }

            # Author(s)
            elif line.startswith('#@'):
                current_paper['Authors'] = line[2:].strip()

            # Year
            elif line.startswith('#t'):
                current_paper['Year'] = line[2:].strip()

            # Conference or Journal Venue
            elif line.startswith('#c'):
                current_paper['Venue'] = line[2:].strip()

            # Paper Index ID
            elif line.startswith('#index'):
                current_paper['Index'] = line[6:].strip()

            # Citation to another paper
            elif line.startswith('#%'):
                current_paper['References'].append(line[2:].strip())

            # Abstract
            elif line.startswith('#!'):
                current_paper['Abstract'] = line[2:].strip()

        # Append the last paper after loop ends
        if current_paper:
            current_paper['References'] = ', '.join(current_paper.get('References', [])) \
                if current_paper.get('References') else None
            data.append(current_paper)

    return data


def convert_to_dataframe(data: List[dict]) -> pd.DataFrame:
    """
    Convert the parsed list of paper metadata into a Pandas DataFrame with standardized columns.

    Args:
        data (List[dict]): List of dictionaries containing paper metadata.

    Returns:
        pd.DataFrame: DataFrame with fixed columns.
    """
    df = pd.DataFrame(data)

    # Ensure all expected columns exist
    required_columns = ['Title', 'Authors', 'Year', 'Venue', 'Index', 'References', 'Abstract']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    return df[required_columns]


def main():
    input_path = 'dataset/DBLPOnlyCitationOct19.txt'
    output_path = 'dataset/citation-cooked.csv'

    # Step 1: Parse the citation file
    parsed_data = parse_dblp_citation_file(input_path)

    # Step 2: Convert to DataFrame
    df = convert_to_dataframe(parsed_data)

    # Step 3: Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned citation data to {output_path}")


if __name__ == "__main__":
    main()
