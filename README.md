# Dialogix Setup Guide

This guide will lead you through the steps to set up the Dialogix environment. Please follow the steps listed below carefully.

## Prerequisites

- pyenv
- Docker

## Installation Steps

### 1. Install Python Version

Use `pyenv` to install the required Python version:

\`\`\`bash
pyenv install 3.10.6
\`\`\`

### 2. Create Virtual Environment

Create a virtual environment specifically for Dialogix:

\`\`\`bash
pyenv virtualenv 3.10.6 dialogix
\`\`\`

### 3. Activate Virtual Environment

Activate the newly created virtual environment:

\`\`\`bash
pyenv activate dialogix
\`\`\`

### 4. Local Environment Settings

Make sure you are in the Dialogix directory and set up the local environment there:

\`\`\`bash
# In the Dialogix directory
pyenv local dialogix
\`\`\`

### 5. Install Dependencies

Install the required dependencies:

\`\`\`bash
pip3 install -r requirements.txt
\`\`\`

### 6. Set Up Elasticsearch with Docker

Pull the Elasticsearch Docker image and start a container:

\`\`\`bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.2
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
\`\`\`

## Completion

To run, cd into streamlit and execute the following command: `streamlit run Home.py`
