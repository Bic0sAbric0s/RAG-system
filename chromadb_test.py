import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()

collection = client.get_or_create_collection(
    name='test',
    metadata={'hnsw:space':'cosine'},
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

document = [
    "Инструкция по сбросу пароля: перейдите на страницу входа и нажмите 'Забыли пароль?'",
    "Наш офис находится в Москве на Тверской улице, дом 15",
    "Компания была основана в 2010 году и специализируется на AI-решениях"
]

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=document,
    metadatas=[
        {"source": "manual.pdf", "topic": "support"},
        {"source": "info.txt", "topic": "contacts"},
        {"source": "history.txt", "topic": "about"}
    ]
)

query = 'Как восстановить доступ к аккаунту?'

result = collection.query(
    query_texts=[query],
    n_results=2
)
print(result['documents'])
