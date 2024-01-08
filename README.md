# Retrieval-Augmented-Generation
RAG is a recent technique in artificial intelligence that combines information retrieval with text generation. It allows large language models (LLMs) to access and incorporate additional information from external sources while generating text, making their reponses more relevant and informative.

![image](https://github.com/vaishaliag08/Retrieval-Augmented-Generation/assets/68223127/bcd19227-9feb-48df-9b84-765d1636c6ab)

General purpose language models can be fine-tuned to achieve several common tasks such as sentiment analysis and named entity recognition. These tasks generally don’t require additional background knowledge. For more complex and knowledge intensive tasks, it’s possible to build a language model-based system that accesses external knowledge sources to complete tasks. This enables more factual consistency, improves reliability of the generated responses and helps to mitigate the problem of "hallucination".

RAG combines an information retrieval component with a text generator model. RAG can be fine-tuned and its internal knowledge can be modified in an efficient manner and without needing retraining of the entire model.

RAG takes an input and retrieves a set of relevant documents given a source (a set of documents). The documents are concatenated as context with the original input prompt and fed to the text generator which produces the final output. This makes RAG adaptive for situations where facts could evolve over time. This is very useful as LLM’s parametric knowledge is static. RAG allows language models to bypass retraining, enabling access to the latest information for generating reliable outputs via retrieval-based generation.

The chatbot created in this project first takes the user documents as input i.e. the external knowledge for factual information, then creates the vector embeddings of the input documents and store them in vector database. Now it's ready to answer any questions related to the documents, the answer that are more factually correct.

https://github.com/vaishaliag08/Retrieval-Augmented-Generation/assets/68223127/728a512c-2672-44d8-a538-01e90312c3a1



