import json
import os
import mlflow
from haystack.document_stores import OpenSearchDocumentStore
from haystack.schema import MultiLabel, Label, Document, Answer
from dotenv import load_dotenv
from haystack.document_stores import ElasticsearchDocumentStore


def evaluate_pipeline(pipeline, retriever, reader_model, pipeline_nodes, evaluation_dataset, evaluation_index,
                      retriever_top_k=5, reader_use_gpu=False):
    """
      Conducts an evaluation of a given Haystack pipeline, logs metrics to MLflow,
      stores results in an OpenSearch document store, and cleans up after execution.

      Args:
          pipeline (Pipeline): The Haystack pipeline object that is to be evaluated.
          retriever (str): Identifier for the retriever component used within the pipeline.
          reader_model (str): Identifier for the reader model used within the pipeline.
          pipeline_nodes (list): A list of pipeline node names in the order they are connected within the pipeline.
          evaluation_dataset (str): File path to the dataset used for evaluation.
          evaluation_index (str): The name of the index in the OpenSearch document store where evaluation results will be stored.
          retriever_top_k (int, optional): The number of top documents to retrieve in the retrieval step. Defaults to 5.
          reader_use_gpu (bool, optional): Flag to indicate whether the reader should use GPU for processing. Defaults to False.

      Returns:
          None. The function's purpose is to execute a side-effect-heavy procedure that logs data to external systems.

      Side Effects:
          Produces log entries in MLflow for parameters and SAS scores.
          Stores evaluation results in both a local JSON file and an OpenSearch index.
          Deletes the local JSON file after its contents have been logged to MLflow.

      Raises:
          IOError: If there is an issue with file reading or writing.
          RuntimeError: If there are issues with connecting to OpenSearch or MLflow.

      Usage Example:
          To use this function, ensure that all environment variables related to OpenSearch and MLflow are set in your environment.
          Then, call the function with the appropriate parameters:

          pipeline = ...  # Set up your Haystack pipeline
          evaluate_pipeline(
              pipeline=pipeline,
              retriever="my_retriever",
              reader_model="my_reader_model",
              pipeline_nodes=["Retriever", "Reader"],
              evaluation_dataset="path/to/dataset.json",
              evaluation_index="my_evaluation_index"
          )
      """

    document_store = ElasticsearchDocumentStore(
        host="localhost",
         username="",
         password="",
         embedding_dim=512,
         recreate_index=True,
         index=evaluation_index)

    # Configure MLflow client and experiment
    # mlflow_tracking_uri = 'http://localhost:5000'

    # mlflow_experiment_name = 'qa-service-logs'


    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    # mlflow.set_experiment(mlflow_experiment_name)

    # with mlflow.start_run():
    #     # Log Parameters

    #     # Log Parameters for the Retriever
    #     mlflow.log_param("retriever", retriever)
    #     mlflow.log_param("retriever_top_k", retriever_top_k)  # Anzahl der zurückgegebenen Dokumente

    #     # Log Parameters for the Reader
    #     mlflow.log_param("reader_model", reader_model)
    #     mlflow.log_param("reader_use_gpu", reader_use_gpu)

    #     # Log Parameters for the Pipeline Configuration
    #     mlflow.log_param("pipeline_nodes", pipeline_nodes)
    #     mlflow.log_param("evaluation_dataset", evaluation_dataset)

        # Loading Evaluation Data
    with open(evaluation_dataset, 'r') as f:
        eval_data = json.load(f)

    eval_labels = [
        MultiLabel(
            labels=[
                Label(
                    query=label['labels'][0]['query'],
                    answer=Answer(
                        answer=label['labels'][0]['answer']['answer'],
                        offsets_in_context=None
                    ),
                    document=Document(
                        id=label['labels'][0]['document']['id'],
                        content_type=label['labels'][0]['document']['content_type'],
                        content=label['labels'][0]['document']['content']
                    ),
                    is_correct_answer=label['labels'][0]['is_correct_answer'],
                    is_correct_document=label['labels'][0]['is_correct_document'],
                    origin=label['labels'][0]['origin']
                )
            ]
        )
        for label in eval_data
    ]

        # Calculating SAS

    advanced_eval_result = pipeline.eval(
        labels=eval_labels, params={"Retriever": {"top_k": 5}}, sas_model_name_or_path="cross-encoder/stsb-roberta-large"
    )

    metrics = advanced_eval_result.calculate_metrics()
    print(metrics)
    sas_score = metrics[pipeline_nodes[-1]]["sas"]
    print(sas_score)
        # mlflow.log_metric("SAS Score", sas_score)

        # # Loading Evaluation Dataset
        # with open(evaluation_dataset, 'r') as f:
        #     gold_data = json.load(f)

        # comparison_results = []
        # for item in gold_data:
        #     question = item['labels'][0]['query']
        #     gold_answer = item['labels'][0]['answer']['answer']

        #     prediction = pipeline.run(query=question)

        #     if prediction['answers']:
        #         generated_answer = prediction['answers'][0].answer
        #     else:
        #         generated_answer = "Keine Antwort gefunden"

        #     comparison_result = {
        #         'Question': question,
        #         'Gold Answer': gold_answer,
        #         'Generated Answer': generated_answer
        #     }
        #     comparison_results.append(comparison_result)

        # comparison_json_path = "comparison_results.json"
        # with open(comparison_json_path, "w") as f:
        #     json.dump(comparison_results, f, ensure_ascii=False, indent=4)

        # # Loading Results into MlFlow
        # mlflow.log_artifact(comparison_json_path)

        # # Loading Results in Evaluation Index
        # documents_to_index = [
        #     {
        #         "content": json.dumps(result),
        #         "meta": {
        #             "Question": result["Question"],
        #             "Gold Answer": result["Gold Answer"],
        #             "Generated Answer": result["Generated Answer"]

        #         }
        #     }
        #     for result in comparison_results
        # ]

        # document_store.write_documents(documents_to_index, index=evaluation_index)

        # # Delete temporary JSON File
        # if os.path.exists("comparison_results.json"):
        #     os.remove("comparison_results.json")
