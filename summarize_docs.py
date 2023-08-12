from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def summarize_docs(docs):
    print("Summarizing documents")
    llm = OpenAI(temperature=0.1)
    prompt_template = """Write a concise bullet point summary of the following, not more than 50 words:
        {text}
        SUMMARY IN FIVE BULLET POINTS:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
    summary_text = chain.run(docs)
    # print("summarize_docs:"+summary_text)
    return summary_text