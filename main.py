from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
from function import *
import os




def llama_evaluate(generate_text, test_df, prompt_num):
    # generate text using llama
    composed_resp = []
    model_resp = generate_response(generate_text, test_df, prompt_num)
    composed_resp.extend(model_resp)
    df_original_resp = pd.DataFrame(composed_resp, columns=['response'])
    print(df_original_resp.head(prompt_num))
    
    label_list = extract_all_labels(test_df['Text'], test_df['Label'], df_original_resp, prompt_num)

    df_label = pd.DataFrame(label_list, columns=['Text', 'response', 'Category', 'Ori_label', 'Format Compliance'])
    file_path = 'Category_Data.csv'
    df_label.to_csv(file_path, index=False)
    
    accuracy, format_accuracy, precision, recall, f1 = evaluate(df_label, test_df)
    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("format_accuracy: ", format_accuracy)
    print("llama finished")
    return df_original_resp
    
def rag_llama_evaluate(prompt_num, test_df, qa):
    composed_rag_resp = []
    rag_resp = rag_response(prompt_num, test_df, qa)
    composed_rag_resp.extend(rag_resp)
    df_rag_resp = pd.DataFrame(composed_rag_resp, columns=['response'])
    rag_label_list = extract_all_labels(test_df['Text'], test_df['Label'], df_rag_resp, prompt_num)

    df_rag_label = pd.DataFrame(rag_label_list, columns=['Text', 'response', 'Category', 'Ori_label', 'Format Compliance'])
    file_path = 'Category_rag_Data.csv'
    df_rag_label.to_csv(file_path, index=False)
    
    accuracy, format_accuracy, precision, recall, f1 = evaluate(df_rag_label, test_df)
    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("format_accuracy: ", format_accuracy)
    print("rag finished")
    return rag_label_list
    
def main(option, data_path):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # load Llama-2 model and tokenizer
    Token = "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"

    llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token = Token)
    model = AutoModelForCausalLM.from_pretrained(llama_model_name, device_map="auto",token = Token, torch_dtype = torch.float16)

    generate_text = transformers.pipeline(
        model= model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        torch_dtype= torch.float16,
        # we pass model parameters here too
        temperature = 1.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens= 200,  # mex number of tokens to generate in the output
        repetition_penalty = 1.1,  # without this output begins repeating
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    
    # number of samples of test_df
    prompt_num = 10
    if(option == 'llama'):
        llama_evaluate(generate_text, test_df, prompt_num)
    elif(option == 'rag'):
        vector_store = create_corpus(train_df)
        qa = retriever_chain(vector_store, llm)
        rag_llama_evaluate(prompt_num, test_df, qa)
    else:
        print("Wrong option")

if __name__ == "__main__":
    # gpu set, you can use the GPU which have enough resource
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # you can change the file path if there's something wrong
    # data_path = r"df_file.csv"
    data_path = r"data/df_file.csv"

    # choose option between llama or rag.
    # main('llama', data_path)
    main('rag', data_path)