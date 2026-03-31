from langchain_community.document_loaders import PyPDFLoader

# ✅ Load PDF (static path)
loader = PyPDFLoader("DOCUMENTLOADERS/Stock_Price_Prediction_Using_Time_Series.pdf")

documents = loader.load()

# ✅ Print content
# print(documents[0].page_content)

print(len(documents))
print(documents[0].metadata)