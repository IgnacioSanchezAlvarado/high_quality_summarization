from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain.output_parsers import PydanticOutputParser

import json
import os
import sys
import boto3
from pydantic import BaseModel, Field
from typing import List
import re

from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_json(text):
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group()
    return None

class DocumentoSecciones(BaseModel):
    sections: List[str] = Field(description="List of document sections")

parser = PydanticOutputParser(pydantic_object=DocumentoSecciones)

def summarize_section(seccion, docs, llm_chat):
    bedrock_runtime = boto3.client('bedrock-runtime')

    llm_chat_2 = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.1},
        client=bedrock_runtime,
        region_name="us-west-2"
    )
    prompt_template = f"""You are an assistant specialized in summarizing documents. Your goal is to help users quickly understand the most important points of a document, including key numerical information, main outcomes, general content etc.
    
    You will be provided with a complete document and you will have to perform a detailed summary of only the section indicated below. For this summary, you can use the context information from the entire document, but the final result must be only the summary of the specific section indicated.
    
    Section to summarize: {seccion}
    Document: "{{text}}"
    
    Important Instructions:
    
    Provide the content of the summary directly, without additional introduction or conclusion.
    Do not repeat the section title in the summary.
    Structure the summary in short and concise paragraphs.
    Use bullets to list key points or important numerical data.
    Make sure to include all relevant and numerical information from the section.
    Maintain a professional and objective tone.
    Do not use phrases like "Summary of the section" or similar.
    Provide the output in the same language as the original document.
    
    Summary:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    llm_chain = LLMChain(llm=llm_chat_2, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    summary_section = stuff_chain.run(docs)
    return seccion, summary_section

def high_quality_summarization(loader):
    bedrock_runtime = boto3.client('bedrock-runtime')

    llm_chat_2 = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.1},
        client=bedrock_runtime,
        region_name="us-west-2"
    )
    ### Extracting the sections from the original document
    prompt_template = """Identify the main sections and subsections of the following document. Include the page number where each section and subsection starts and ends. Provide the output in the same language as the original document:
    <example>
    "Section 1: Title of Section 1 (pages 1-5)",
    "Section 2: Title of Section 2 (pages 6-10)",
    .... rest of sections
    </example>
    "{text}"
    List of sections and subsections:"""
        
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Define LLM chain
    
    llm_chain = LLMChain(llm=llm_chat_2, prompt=prompt)
    
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    docs = loader.load()
    sections = stuff_chain.run(docs)
    
    ### Creating the sections list
    prompt_template = """You will be provided with a list of sections from a document. Create a detailed list that includes all sections and subsections, with their respective page ranges.

    Important Instructions:
    
    Prioritize subsections over general sections.
    If a general section is completely divided into subsections, omit the general section.
    If there are parts of a general section not covered by subsections, include only those parts.
    For sections without specific numbering, use 'N/A' as the page range.
    If you are unsure of the page where a section ends, indicate that it ends on the starting page of the next section.
    {format_instructions}
    
    <sections> {text} </sections>
    Make sure the output complies with the specified format and follows the given instructions. Provide the output in the same language as the original document.
    """
    
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Define LLM chain
    llm_chain = LLMChain(llm=llm_chat_2, prompt=prompt)
    
    # Run the chain
    resultado = llm_chain.run(sections)
    
    try:
        json_str = extract_json(resultado)
        if json_str:
            parsed_json = json.loads(json_str)
            section_list = parsed_json.get('sections', [])
        else:
            raise ValueError("Not valid JSON in the response")
    except json.JSONDecodeError as e:
        print(f"Deconding JSON error: {e}")
        print("Original output:", resultado)
        section_list = []
    except Exception as e:
        print(f"Processing error: {e}")
        print("Original output:", resultado)
        section_list = []
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(summarize_section, seccion, docs, llm_chat_2) for seccion in section_list]
        
        # Collect results in order
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    # Sort results to maintain original order
    results.sort(key=lambda x: section_list.index(x[0]))
    
    # Combine results with improved formatting
    output_final = ""
    for seccion, summary in results:
        output_final += f"## **{seccion}**\n\n{summary.strip()}\n\n---\n\n"
    
    return sections, json_str, section_list, output_final