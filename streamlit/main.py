from haystack.document_stores import ElasticsearchDocumentStore
from haystack.schema import Document
from utility import *
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline
from haystack.nodes import EmbeddingRetriever
#from evaluation import evaluation_pipeline
import pandas as pd
import os
import re
import csv
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader, SentenceTransformersRanker
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory
from haystack.pipelines import Pipeline
import requests
import json
import os
import re
import csv
from haystack.schema import Document
import pandas as pd
from openai import OpenAI
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline
from haystack.nodes import EmbeddingRetriever

testQuery = "Did something unusual happen considering SSH Security?"

open_ai_api_key = "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"

rag_prompt = PromptTemplate(
    prompt="""Create a comprehensive answer to the question based on the following text. Summarize key points from the text to provide a clear and concise response. Your answer should be formulated in your own words and not exceed 50 words.

    \n\n Associated Text: {join(documents)} \n\n Question: '{query}' \n\n Answer:""",
    output_parser=AnswerParser()
)

# This list will hold pairs of (query, response)
conversation_history = []

def setup(filename):
    """
    Sets up document store from clean templates file and returns a retriever.

    Param:
    filename: Filename (stem) of target file.

    Returns:
    Retriever to be used in queries.
    """
    # Open previously cleaned templates file
    with open(input_dir + filename + '_clean_templates.csv', 'r') as file:
        logs = file.readlines()

    # Add each cluster as a document by content, using doc_id and frequency as metadata
    documents = []
    for row in logs:
        if row:
            split_values = row.split(',')
            documents.append(Document(content=split_values[2], meta={'id': split_values[0], 'frequency': split_values[1]}))

    # Build document store
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="your_index_name",
        embedding_dim=512,
        recreate_index=True)
    document_store.write_documents(documents)

    # Build retriever
    retriever = EmbeddingRetriever(document_store=document_store,
     embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1")
    document_store.update_embeddings(retriever)
    return document_store, retriever

def query_handler(query, retriever, filename, top_k=5, freq_lim=10):
    retrieved_docs = retriever.retrieve(query=query, top_k=top_k) # Retrieve top matching templates according to query

    meta_dict = {}
    print(retrieved_docs)

    # Build metadata dictionary mapping id to frequency
    for doc in retrieved_docs:
        doc_id = doc.meta['id']
        frequency = doc.meta['frequency']
        meta_dict[doc_id] = frequency

    # Filter content corresponding to retrieved templates and present to GPT as string
    filtered_content = ""
    df_logs = pd.read_csv(input_dir + filename + '_clean_structured.csv', names=['doc_id', 'content'], engine="python", quotechar=None, quoting=3)
    if len(meta_dict) > 1:
        for doc_id, freq in meta_dict.items():
            freq = int(freq)
            rows = df_logs[df_logs['doc_id'] == doc_id]
            if freq > freq_lim:
                rows = rows.head(freq_lim)
            filtered_content += ', '.join(rows['content'].tolist())


        return filtered_content
    return ""

def initialize_pipeline(document_store, prompt_node):
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1")
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="Reader", inputs=["Retriever"])
    return pipe

# Function to Process Initial Query
def process_initial_query(query, pipeline):

    # Run the pipeline with the query
    output = pipeline.run(query=query)



    # Retrieve the answers from the pipeline output
    answers = output.get('answers', [])
    return [answer.answer for answer in answers]


prompt_node_2 = PromptNode(model_name_or_path="text-davinci-003", api_key=open_ai_api_key, default_prompt_template=None)

def get_gpt_response(prompt):
    client = OpenAI(api_key="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")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a expert in Reading Linux Logs"},
        {"role": "user", "content": prompt}
        ]
    )

    response = completion.choices[0].message.content
    print(response)
    return response




# Function to Process Conversational Query
def process_conversational_query(query, pipeline, context=None):
    # Run the pipeline with the new query
    pipeline_output = pipeline.run(query=query)
    pipeline_answers = pipeline_output.get('answers', [])
    pipeline_answer_text = " ".join([answer.answer for answer in pipeline_answers])
    if query != 'full logs':
        gpt_prompt = "Then based on the current user question: "+ query + " and answer of our pipeline: " + pipeline_answer_text + ", formulate a response. The response shouldn't be longer than 2 sentences. Here is the conversation history:\n"
        for q, a in conversation_history:
            gpt_prompt += f"Q: {q}\nA: {a}\n"


        gpt_response = get_gpt_response(gpt_prompt)

        # Update the conversation history
        conversation_history.append((query, gpt_response))

        return gpt_response
    else:
        return [context]


def initialize(filename):
    # Preprocessing
    preprocessing(filename)
    clean_template_data(filename)
    clean_structured_data(filename)
    #initial_query = input("Enter initial query: ")
    return

def initial_query_handler(initial_query, retriever, document_store, filename):
    context = query_handler(initial_query, retriever, filename, top_k=5, freq_lim=10)

    new_prompt = PromptTemplate(
        prompt="""Analyze the provided context focusing on dates and specific events to answer the question with precision. Use the filtered information to detail how certain events occurred and mention relevant dates. Your response should be direct, clearly referencing specific incidents and dates from the context. Your response is limited to 50 words, unless you are asked to provide the full logs corresponding to a certain event.

        \n\nContext: '{}' \n\nQuestion: '{{query}}' \n\nDetailed Answer:""".format(context),
        output_parser=AnswerParser()
    )

    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=open_ai_api_key, default_prompt_template=new_prompt)

    pipeline = initialize_pipeline(document_store, prompt_node)
    initial_results = process_initial_query(initial_query, pipeline)
    conversation_history.append((initial_query, " ".join(initial_results)))

    print("\nInitial Query Results:")
    for result in initial_results:
         print(result)
    return initial_results, pipeline, context

def follow_up_handler(follow_up_query, pipeline, context=None):
    if follow_up_query.lower() == 'exit':
        return None

    answer = process_conversational_query(follow_up_query, pipeline, context)
        # Here you'd call GPT-3.5 with chatgpt_prompt to get the response
        # Since the API call isn't implemented here, print the prompt for now
    return answer



def gptQuery(context, retriever):
    #BUILDING THE GPT READER

    new_prompt = PromptTemplate(
        prompt="""Analyze the provided context focusing on dates and specific events to answer the question with precision. Use the filtered information to detail how certain events occurred and mention relevant dates. Your response should be direct, clearly referencing specific incidents and dates from the context, and limited to 50 words.

        \n\nContext: '{}' \n\nQuestion: '{{query}}' \n\nDetailed Answer:""".format(context),
        output_parser=AnswerParser()
    )

    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=open_ai_api_key, default_prompt_template=new_prompt)



    # SETTING UP PIPELINE
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="Reader", inputs=["Retriever"])


    # RUNNING THE PREDICTION
    prediction = pipe.run(
        query="Did something unusual happen considering SSH Security? Look for anomalies, look for things that could be the issue of a problem",
        params={
            "Retriever": {"top_k": 5}  # Include context here
        }
    )

    answers = [ans.answer for ans in prediction['answers']]


    """evaluate_pipeline(pipe,
                      retriever="sentence-transformers/distiluse-base-multilingual-cased-v1",
                      reader_model="GPT DAVINCI",
                      evaluation_dataset="evaluation_data.json",
                      evaluation_index="evaluation_index",
                      pipeline_nodes=["Retriever", "Reader"])"""
    print(answers)
    return
