#!/bin/bash
# Directory containing the text files
directory="."
# Loop through each text file in the directory
for file in "$directory"/*.txt; do
# Check if the file exists
if [ -e "$file" ]; then
# Acquire a lock and execute the Python script
{
flock -x 200
python3 parse_abnv.py "$file"
} 200>"$file.lock" # Use a separate lock file for each text file
fi
done
