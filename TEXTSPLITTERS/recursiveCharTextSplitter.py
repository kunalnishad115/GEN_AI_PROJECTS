from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

docs=PyPDFLoader('TEXTSPLITTERS/Stock_Price_Prediction_Using_Time_Series.pdf').load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

texts=text_splitter.split_documents(docs)

# print(texts)

print(f"\nTotal chunks: {len(texts)}")