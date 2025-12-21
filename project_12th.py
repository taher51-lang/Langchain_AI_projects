from dotenv import load_dotenv
from typing import Literal,Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
import streamlit as st
from pydantic import BaseModel,Field
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
memory_unit = [SystemMessage("You are an emotional Support AI assistant")]
class Analyze(BaseModel):
    userName: Optional[str] = Field(default=None, description="The name of the user if mentioned, else null") 
    userMood: Literal["Very Good", "Good", "Bad", "Stressed","Sad"] = Field(
        description='''Categorize the mood of the user into exactly one of these options: "Very Good" If the user is seems very excited
        "Good" if user is seems a little upset. "Bad" if user seems really upset "Sad" if user is sad about some topic
        "stressed" if user is stressing and has tension '''
    )
    userAbout: str = Field(description="A 5-word summary of what the user is talking about")

llm_ = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm_)
parser = PydanticOutputParser(pydantic_object=Analyze)

prompt1 = PromptTemplate(
    template="You are a helpful assistant. Wrap your response in JSON format.\n{format_instructions}\nUser Input: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='''Based on the following output:
    userName:{userName},userMood:{userMood},userAbout:{userAbout}.
    History{History}
    Generate a 2 line supportive output
    ''',
    input_variables=["userName","userMood","userAbout","History"]
)
prompt3 = PromptTemplate(
    template='''Summarize the following chat history into a concise 150-word paragraph. 
    DO NOT repeat the dialogue. 
    DO NOT use 'Human:' or 'AI:' tags in your summary.
    Extract only key facts (like names, colors, and feelings) and the current status of the conversation.{text}''',
    input_variables=['text']
)
# Method for Data transfer, used this instead of Lambda
def Prepare_data_(analyze_obj,History):
    return {
        "userName":analyze_obj.userName,
        "userMood":analyze_obj.userMood,
        "userAbout":analyze_obj.userAbout,
        "History": History}
parser_str = StrOutputParser()
analyze_chain = prompt1 | model | parser
support_msg_chain = prompt2 | model | parser_str
# History = str(memory_unit).replace("[","").replace("]","").replace('"'," ")
while True:
    History = ""
    inp = input("Hey! How was your day? (q to quit): ")
    if inp.lower() == "q":
        break
    pyObj = analyze_chain.invoke({"text": inp})
    memory_unit.append(HumanMessage(content=inp))    
    History = "" # This prevents the 'snowball' effect
    for msg in memory_unit:
        # Using .content avoids the additional_kwargs/metadata junk
        if isinstance(msg, SystemMessage):
            History += f"Context: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            History += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            History += f"AI: {msg.content}\n"
    result = support_msg_chain.invoke({
        "userName": pyObj.userName,
        "userAbout": pyObj.userAbout,
        "userMood": pyObj.userMood,
        "History": History  # Now this contains the message you just typed!
    })
    memory_unit.append(AIMessage(content=result))
    if len(memory_unit)>5:
        chain = prompt3 | model | parser_str
        result_summary = chain.invoke({"text":History})
        memory_unit.clear()
        memory_unit.append(SystemMessage(content=result_summary))
        print("--summary--")
        print(History)
    print(f"\nAI: {result}\n")

    
    


