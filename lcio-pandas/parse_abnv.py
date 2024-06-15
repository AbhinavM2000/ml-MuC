import argparse
import pandas as pd
import os

def find_line_number(file_path, target_text):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if target_text in line:
                return i + 1  # Line numbers start from 1
    return None

def clean_temp_file(temp_file):
    with open(temp_file, 'r') as file:
        content = file.read()

    # Remove unwanted spaces and characters from each line
    content = content.replace('->', '').replace('(', '').replace(')', '').replace('|', ',')

    # Remove leading and trailing spaces from each line
    content = '\n'.join(line.strip() for line in content.split('\n'))

    with open(temp_file, 'w') as file:
        file.write(content)

def copy_lines(source_file, destination_file, start_line, end_line, heading):
    with open(source_file, 'r') as source, open(destination_file, 'a') as dest:
        dest.write(heading + '\n')
        lines = source.readlines()[start_line - 1:end_line - 1]
        for line in lines:
            if line.strip().startswith('->'):
                dest.write(line)

def process_file(filename):
    # Remove spaces from all lines and save the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    lines = [line.replace(' ', '') for line in lines]
    with open(filename, 'w') as file:
        file.writelines(lines)

    # Open the file again to perform further operations
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find line numbers for different collections
    a = lines.index(next(line for line in lines if 'ECalBarrelCollection' in line))
    b = lines.index(next(line for line in lines if 'ECalEndcapCollection' in line))
    c = lines.index(next(line for line in lines if 'HCalBarrelCollection' in line))
    d = lines.index(next(line for line in lines if 'HCalEndcapCollection' in line))

    # Initialize a pandas dataframe
    df = pd.DataFrame(columns=['CollectionID', 'PPDG', 'Energy', 'Time', 'Length', 'SPDG', 'X', 'Y', 'Z'])

    # Read CSV values and append to the dataframe
    data_frames = []
    for start, end, prefix in [(a+3, b-1, 'ECBC'), (c+3, d-1, 'ECEC'), (d+3, len(lines)-1, 'HCBC'), (c+3, d-1, 'HCEC')]:
        data = []
        for line in lines[start:end]:
            values = line.strip().split(',')
            data.append([prefix] + values)
        temp_df = pd.DataFrame(data, columns=df.columns)
        data_frames.append(temp_df)
    df = pd.concat(data_frames, ignore_index=True)

    # Print dataframe information
    print(df.info())
    # Convert dataframe columns to numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    return df



def main():
    parser = argparse.ArgumentParser(description='Process input file and generate output in .abnv format.')
    parser.add_argument('input_file', metavar='input_file', type=str, help='Path to input file')
    args = parser.parse_args()

    input_file = args.input_file
    temp_file = 'temp.txt'

    ecal_barrel_start = find_line_number(input_file, 'ECalBarrelCollection')
    ecal_endcap_start = find_line_number(input_file, 'ECalEndcapCollection')
    hcal_barrel_start = find_line_number(input_file, 'HCalBarrelCollection')
    hcal_endcap_start = find_line_number(input_file, 'HCalEndcapCollection')
    hcal_ring_start = find_line_number(input_file, 'HCalRingCollection')

    copy_lines(input_file, temp_file, ecal_barrel_start, ecal_endcap_start, 'ECalBarrelCollection')
    copy_lines(input_file, temp_file, ecal_endcap_start, hcal_barrel_start, 'ECalEndcapCollection')
    copy_lines(input_file, temp_file, hcal_barrel_start, hcal_endcap_start, 'HCalBarrelCollection')
    copy_lines(input_file, temp_file, hcal_endcap_start, hcal_ring_start, 'HCalEndcapCollection')

    clean_temp_file(temp_file)  # Clean up the temp file after copying and before finishing

    filename = 'temp.txt'
    df = process_file(filename)
    print(df.head())

    # Print rows where PPDG is negative
    print(df[df['PPDG'] < 0])

    # Save the dataframe to a .abnv file in csv format
    output_file = os.path.splitext(input_file)[0] + '.abnv'
    df.to_csv(output_file, index=False)

    # Remove the temporary file
    os.remove(temp_file)

if __name__ == "__main__":
    main()
