from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# ✅ Load PDF
loader = PyPDFLoader('TEXTSPLITTERS/Stock_Price_Prediction_Using_Time_Series.pdf')
docs = loader.load()

# ✅ Split
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator="\n"
)

texts = text_splitter.split_documents(docs)

# ✅ Output
# print(texts[0].page_content)
print(f"\nTotal chunks: {len(texts)}")