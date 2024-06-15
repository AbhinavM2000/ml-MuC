#!/bin/bash
# Counter for the output file names
count=0
# Loop through all .slcio files in the current directory
for file in *.slcio; do
# Check if the file exists
if [ -e "$file" ]; then
# Run the command for each file and redirect the output to a text file
dumpevent "$file" 1 > "eventdump$count.txt"
# Increment the counter
((count++))
fi
done
