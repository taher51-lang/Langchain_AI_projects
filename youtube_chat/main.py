from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

# Phase 1 of RAG Implementation(Aquiring Knowledge Source)
yt_api = YouTubeTranscriptApi()
try:
    fetchList = yt_api.fetch(video_id="E4l91XKQSgw",languages=["en"])
    print(type(fetchList))
    text = ""
    for snippet in fetchList:
        text+=snippet.text
    print(len(text))
except:
    print('Transcript Unavailable')
# Phase 2 Text Splitting using Recursive Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
chunk_list2 = splitter.create_documents([text])
# for i in chunk_list:
#     print(i)
# Phase 3: Embedding Generation and storage in database
load_dotenv()
llm = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunk_list2,embedding=llm)
# Phase 4 L Retriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":4})
# Injecting the context and query into the llm
prompt = PromptTemplate(
    template='''You are a Helpful assistant
    ONLY from the given Context : {context}, answer the following question:{question},
    If the question is not related to context, SAY I DONT KNOW''',
    input_variables=["context","question"]
)
def format(retrivedDocs):
    cn = "\n\n".join(doc.page_content for doc in retrivedDocs)
    return cn
parallelChain = RunnableParallel({
    "context": retriever | RunnableLambda(format),
    "question":RunnablePassthrough(),
}
)
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)
# result = retriever.invoke("What is ollama")
# print(format(result))
parser = StrOutputParser()
main_chain = parallelChain | prompt | llm | parser 
result = main_chain.invoke("what is ollama")
print(result)