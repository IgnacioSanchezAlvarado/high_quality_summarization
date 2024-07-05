from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
import boto3


def regular_summary(loader):
    bedrock_runtime = boto3.client('bedrock-runtime')

    llm_chat_2 = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.1},
        client=bedrock_runtime,
        region_name="us-west-2"
    )
    
    prompt_template = """Write a long and complete summary of the following:
    "{text}"
    SUMMARY:"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    
    llm_chain = LLMChain(llm=llm_chat_2, prompt=prompt)
    
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    docs = loader.load()
    output = stuff_chain.run(docs)
    return output
    