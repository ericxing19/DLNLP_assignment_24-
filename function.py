import re
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import pandas as pd
import numpy as np

def get_prompt(text):
    prompt = f"""  A <<<Text>>> will be provided, you are GREAT at analysing the <<<TEXT>>> to classify it into different categories.\n In total, there are 5 categories: Politics = 0, Sport = 1, Technology = 2, Entertainment =3, Business = 4"\n
    
    You need to directly classify the <<<Text>>> provided into one of the five categories, which means you answer MUST and CAN ONLY be {"Category: 0/1/2/3/4"}. After giving the category answer, you can provide some reasons for your choice.
    
    <<<<<start example:
    
    Text: Budget to set scene for election

    Gordon Brown will seek to put the economy at the centre of Labour's bid for a third term in power when he delivers his ninth Budget at 1230 GMT. He is expected to stress the importance of continued economic stability, with low unemployment and interest rates. The chancellor is expected to freeze petrol duty and raise the stamp duty threshold from Â£60,000. But the Conservatives and Lib Dems insist voters face higher taxes and more means-testing under Labour.
    
    Treasury officials have said there will not be a pre-election giveaway, but Mr Brown is thought to have about Â£2bn to spare.

    Answer:{"Category: 0"};

    example End>>>>>
    
    <<<Text Start: {text} \n  Text End>>
    
    """
    return prompt

def generate_response(generate_text, test_df, prompt_num):
    model_resp = []
    for i in range(prompt_num):
        prompt = get_prompt(test_df['Text'].iloc[i])
        
        output = generate_text(prompt)
        # print("output: ", output)
        # trim the output by removing prompt
        model_response = output[0]['generated_text']
        # print("model response: ", model_response)
        trimmed_output =model_response[len(prompt):]
        
        # collecting responses
        model_resp.append(trimmed_output)

        print(f"the",(i+1),"th prompt responded")
        
    return model_resp

def extract_labels(text):
    # label patterns
    pattern1 = r"Category:\s(\d*)"
    # create match
    match1 = re.search(pattern1, text)
    # find match
    label1 = match1.group(1) if match1 else None

    # give a format marker , 1 if classification label missing and 0 if not 
    label_format = 1 if label1 is None else 0
    
    return label1, label_format

def extract_all_labels(text, ori_label, response_list, num):
    label_list = []
    for i in range(num):
        resp = response_list['response'][i]
        label1, label_format = extract_labels(resp)
        label_list.append([text.iloc[i], resp, label1, ori_label.iloc[i], label_format])
    return label_list

# evaluate
def evaluate(response_df, prompt_df):
    # fill nonetype with -1
    response_df['Category'] = response_df['Category'].replace('', -1).fillna(-1).astype(int)
    
    # Extract predicted labels and actual labels
    y_pred = np.array(response_df['Category'], dtype = int)
    y_true = np.array(prompt_df['Label'], dtype = int)[0:10]
    print(y_pred)
    print(y_true)
    
    # Calculate accuracy
    accurate = sum(y_pred == y_true)
    accuracy = accurate / len(response_df)

    # Calculate format compliance accuracy
    format_accuracy = (1 - response_df['Format Compliance'].sum() / len(response_df)) * 100

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, format_accuracy, precision, recall, f1

def process_document(train_df):
    documents = []
    for i in range(train_df.shape[0]):
        query = train_df.iloc[i]['Text']
        if len(query) > 1200:
            query = query[:1200]
        document = f"Text: {query}, \n Answer: {{Category: {str(train_df.iloc[i]['Label'])}}}"
        documents.append(document)
    return documents

def create_corpus(train_df):
    client = chromadb.Client()
    # if exist,delete
    if "document" in [col.name for col in client.list_collections()]:
        client.delete_collection("document")
        print("deleted")

    collection = client.create_collection("document")

    documents = process_document(train_df)


    # select embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)


    # embed document into vector
    document_embeddings = embedding_model.encode(documents)

    # insert vector into ChromaDB
    for i, embedding in enumerate(document_embeddings):
        collection.add(ids=[str(i)], documents=[documents[i]], embeddings=[embedding.tolist()])

    # initialize HuggingFace Embeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # create Chroma vector store
    vector_store = Chroma(
        collection_name="document",
        embedding_function=hf_embeddings,
        client=client 
    )
    return vector_store

def retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 

    # Create Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer.
    {context}

    Question: 
    A <<<Text>>> will be provided, you are GREAT at analysing the <<<TEXT>>> to classify it into different categories.\n In total, there are 5 categories: Politics = 0, Sport = 1, Technology = 2, Entertainment =3, Business = 4"\n

    You need to directly classify the <<<Text>>> provided into one of the five categories, which means you answer MUST and CAN ONLY be ["Category: 0 (Politics)/1 (Sport)/2 (Technology)/3 (Entertainment)/4 (Business)"]. After giving the category answer, you can provide some reasons for your choice.

    <<<<<start example:

    Text: Budget to set scene for election

    Gordon Brown will seek to put the economy at the centre of Labour's bid for a third term in power when he delivers his ninth Budget at 1230 GMT. He is expected to stress the importance of continued economic stability, with low unemployment and interest rates. The chancellor is expected to freeze petrol duty and raise the stamp duty threshold from Â£60,000. But the Conservatives and Lib Dems insist voters face higher taxes and more means-testing under Labour.

    Answer:["Category: 0"];

    example End>>>>>

    <<<Text Start: {question} \n  Text End>> 

    """

    print(template)

    prompt = PromptTemplate.from_template(template)

    # Initialise RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa

#generate rag_improved response
def rag_response(prompt_num, test_df, qa):
    print(prompt_num)
    rag_resp = []
    for i in range(prompt_num):
        prompt = test_df['Text'].iloc[i]
        if len(prompt) > 1000:
            prompt = prompt[:1000]
        output = qa({"query":prompt})
        # print("output: ", output.keys())
        # trim the output by removing prompt
        # query = output['query']
        # print("query: ", query)
        model_response = output['result']
        # print("model response: ", model_response)
        # find the length of the document added. This helps to find the final result.
        document_length = model_response.find(prompt[:20])
        # print(document_length)
        
        trimmed_output = model_response[(document_length + len(prompt) + 16):]
        
        # collecting responses
        rag_resp.append(trimmed_output)

        print(f"the",(i+1),"th prompt responded")
        
    return rag_resp

