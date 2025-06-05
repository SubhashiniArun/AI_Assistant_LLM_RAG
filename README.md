# Agentic-NotionAI-RAG

### Tools & Technologies:
* Using FTI Architecture
    - F : Foundation Model (LLama)
    - T : Fine-Tuning (Comet, Unsloth)
    - I: Deploy the model as "Inference" endpoint (Hugging Face)
* Using MLOps best practices: Data registries, Model registries, Experiment trackers
* AI optimized web crawling and scrapping using Crawl4AI. Crawling 700+ links and normalizing them as markdown format so that it can be easily ingested into the model [crawl4AI](https://docs.crawl4ai.com/?utm_source=chatgpt.com)
    - test the crawler by running `python test_crawler.py`
* Computing *quality scores* using LLMs
    - Traditional Metrics
        * ROUGE: overlap of n-grams, used in summarization
        * BLEU: precision of the n-gram overlap, used in translation
        * BERTScore: semantic similarity via BERT embeddings, captures the meaning of the n-grams
    - Using LLMs
        * LLM-as-a-judge: LLMs scores the summaries for correctness, fluency, relevance, coherence and conciseness
* Generating *summarization datasets* using *distillation*
    - Summarization datasets: collection of text documents paired with human writen summaries. It could be in JSON format, CSV or extracted from Hugging Face datasets
        `
        {
            "id": "unique id",
            "text_document": "Full input text",
            "summary": "Human written summary"
        }
        `
    - There are two types of summarization datasets
        * Extractive datsets: Selects key sentence directly from the source. Example: Sentence selection
        * Abstractive: Generates new sentence from the source. Example: LLM
    - Distillation: uses powerful models to automatically create high quality (text_document, summary) pairs. Filters out poorly generated summaries, avoids costly human annotation by creating new datasets (text_document, summary pairs)
* Fine tuning the Llama model using Unsloth and evaluate the fine tuned model with COMET
    - Unsloth
        * Fine-tunes the models like Llama 2, Mistral
        * Fasters training time and reduce memory usage
        `uv pip install --no-deps git+https://github.com/unslothai/unsloth.git`
    - COMET
        * Evaluates the fine tuned model
        * Used for summarization to evaluate semantic correctness and fluency
        `uv pip install git+https://github.com/Unbabel/COMET.git`
* Deploying the Llama model as inference endpoint to Hugging Face serverless dedicated endpoints
    - Gives a url once the fine tuned models is deployed
    - Call the url using a post request
* Implementing RAG algorithms using contextual retrieval, hybrid search, MongoDB vector search
    - contextual retrieval: tailoring the retrieval process based on the user's query, improves the relevance of the documents passed to the LLM
    - hybrid search: combination of retrieval methods like dense vector search and sparse keyword search
        * Dense vector match: good for synonyms (semantic match), represented as embeddings
        * Sparse keyword search: good for precise word match, represented as keyword frequency
    - MongoDB vector search: retrieves semantically relevant documents based on the vector embeddings. Example "AI is good" -> vector: [0.123, 0.456, 0.789]
* Uses multiple tools using Hugging Face's smolagents framework
    - smol-agents: An agent (python framework) that interprets code, writes & executes code, uses tools like api/DBs/functions to complete goals autonomously
* Uses LLMOps best practices: prompt monitoring & RAG evaluation 
    - Prompt Monitoring: 
        * Capture and log prompts and model performances
        * Track the performance
        * Detect the hallucinations
        * Audit prompts for compliance and safety
        * Improve the performance quality via analysis
    - RAG evaluation
        * Evaluating the RAG using Opik
        * Opik - Comet's open source platform to evaluate, monitor and optimize the applications (RAG pipelines). Assess both the retrieval and generation components of RAG systems
* Integrate pipeline orchestration, artifact, metadata tracking using ZenML 
    - ZenML
        * Orchestrates RAG pipeline by turning the workflow into modular, traceable and reproducible components (@pipeline, @step)
        * Tracks inputs/outputs for each step
        * Runs on different orchestrators like Kubernetes, Airflow
* Manage the python project using uv and ruff
    - uv
        * dependency resolution
        * uses PEP 582-style
        * generates requirements.txt and lockfiles
        `pip install uv`
    - ruff
        * Python Linter, formatter amnd import sorter
        * PEP8 style
        * `ruff check .` and `ruff format .`
* Apply Software Engineering best practices
