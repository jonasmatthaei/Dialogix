#!/usr/bin/env python3

import sys
import pandas as pd
import re
from logparser.Brain import LogParser

#Parameters for preprocessing
prepro_dataset    = 'HDFS'
input_dir  = './data/' # The input directory of log file
output_dir = './data/'  # The output directory of parsing results
#logfile_name = 'raw_log'
prepro_log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # log format
prepro_regex      = [ # RegEx for preprocessing
    r'blk_(|-)[0-9]+' , 
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', 
]
prepro_threshold  = 2  # Similarity threshold
prepro_delimiter  = []  # Depth of all leaf nodes

# Set up parser with provided parameters
parser = LogParser(logname=prepro_dataset, log_format=prepro_log_format, indir=input_dir, 
                   outdir=output_dir, threshold=prepro_threshold, delimeter=prepro_delimiter, rex=prepro_regex)


def preprocessing(filename='raw_log'):
    """
    Reads raw log data from a file in the ./data directory and clusters it. 
    Produces two files, one containing a list of clusters (defined by a certain pattern) together with their size/frequency.
    The second file is a structured log file, containing all the raw log file information but with each event coupled to a cluster.
    First file: <filename>_templates.csv
    Second file: <filename>_structured.csv

    Args: 
    filename (str): Filename of raw log data.

    Returns: Void
    """
    parser.parse(filename)
    return

def clean_template_data(filename='raw_log'):
    """
    Reads template data from a file, cleans it, and returns a DataFrame with doc_id, frequency and content.

    Args:
    filename (str): Filename of raw log data.

    Returns: Void
    """

    # Read list of clusters
    with open(input_dir + filename + '_templates.csv', 'r') as file:
        logs = file.readlines()

    # Write to file for storage
    with open(output_dir + filename + '_clean_templates.csv', 'w') as file:
        for row in logs:
            if row:  # Check if the row is not empty
                split_values = row.split(',')
                doc_id = split_values[0]
                frequency = split_values[-1].strip()
                content = ' '.join(split_values[1:-1]).strip()
                pattern = r'[\[\]{}*\'""<>()]'
                content = re.sub(pattern, ' ', content)
                content = re.sub(r"\s+", " ", content)
                content = content.strip()
                file.write(', '.join((doc_id, frequency, content)) + "\n") # Add comma-separated doc_id and content to each line
    return

def clean_structured_data(filename='raw_log'):
    """
    Reads structured data from a file, cleans it, and returns a DataFrame with doc_id and content.

    Args:
    filename (str): Filename of raw log data.

    Returns: Void
    """

    # Read list of clusters
    with open(input_dir + filename + '_structured.csv', 'r') as file:
        logs = file.readlines()

    # Write to file for storage
    with open(output_dir + filename + '_clean_structured.csv', 'w') as file:
        for row in logs:
            if row:  # Check if the row is not empty
                split_values = row.split(',')
                doc_id = split_values[-2]
                content = ' '.join(split_values[1:-2]).strip()
                pattern = r'[\[\]{}*\'""<>()]'
                content = re.sub(pattern, ' ', content)
                content = re.sub(r"\s+", " ", content)
                content = content.strip()
                file.write(', '.join((doc_id, content)) + "\n") # Add comma-separated doc_id and content to each line
    return
