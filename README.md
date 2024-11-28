# Course Assistant Bot
### Developed by Reggie Bain
## Overview
#### Background
This project is a proof of concept of what we envision as a broad class of AI-based assistant for course development. It focuses on a tool for building effective course syllabi, which serve a variety of key functions in course development. MIT's Teaching and Learning Lab (referencing Slattery and Carlson (2005)), for example, outlines 3 goals for a syllabus include the document being (1) motivational, (2) structural, and (3) evidentiary. The syllabus can not only serve to outline the key components of the course, but also serves as a sort of "contract" between the students and the instructional staff. In recent years, the rise of AI use in educational contexts (and resulting issues with academic integrity violations) has made the contract aspect of syllabi particularly important. Course policies must be enforced equitably and worded clearly, particularly in [cases where litigation could be involved](https://apnews.com/article/high-school-student-lawsuit-artificial-intelligence-8f1283b517b2ed95c2bac63f9c5cb0b9). This tool is an example of a simple suite of tools that can help instructors, departments, or colleges to quickly assess, among other course resources, how effective syllabi are at answering student questions by actually simulating asking those questions using Large Language Models (LLMs).
#### KPIs
1. Build Retrieval Augmented Generation (RAG) pipeline for asking questions of syllabi
2. Assess/evaluate the pipeline using 2 methods (1) LLMs + Synthetically generated Q&A set and (2) By hand generated Q&A set where the answers are known.
3. Construct basic app that effectively allows users to query documents.
#### Stakeholders
1. Instructors, departments, and colleges that have incentive for courses to have effectively constructed syllabi
2. Students, who will be better able to understand expectations of them (grading policies, resources needed) in courses.
3. Administrators at universities and/or legal teams who want to assess syllabi from a "contract" point of view.
[Click here for write up](https://docs.google.com/document/d/1TLx1REQPteNJ01rnNXXXq0XGCFc9cZoriAiHYC5Np24/edit#heading=h.fcffuc5owalc)
#### Modeling Approach
[Click here for write up](https://docs.google.com/document/d/16rt5T4E6p_cVWof3mqoyO5XgBXt_k8VFKqReElIY_DQ/edit?usp=sharing)
## Data
Our RAG pipeline was tested on several real course syllabi from college-level courses. Although we would have liked to test on a wide variety of types of courses in different subjects and formats (we only had permission to use a physics syllabus from a residential STEM high school and a computer science course at Northwestern University), the pipeline should easily generalize to any reasonably formatted syllabus (or other course document/resource). [In our document store](./documents), you'll find two different formats for a syllabi, one in Markdown and one as a PDF. The pipeline works robustly for both of these formats, but should also work for HTML format. You'll also find JSON files of key sets of questions we tested on each syllabus where we knew the answers were contained in the documents. We explore this more below.
## Embeddings
We used the General Text Embeddings model from HuggingFace's library (https://huggingface.co/thenlper/gte-small) using the small version for practical purposes. This model creates vector embeddings of the document, which is split into a number of chunks. First, we used LangChain and the GTE-Small model to create splits after tokenizing. Atlhgouh the easiest way to split a document into chunks by a simple character count/chunk size, using a tokenizer-based splitter allows for a more semantically relevant division of the document into relevant sections/paragraphs/sentences. Many different models could be used for creating embeddings, but this model was generally recommended as high-performing for open-source, small models. 
## Retrieval
## LLMs
## Results
## Future Work
#### Open Source vs. Proprietary LLMs, APIs
There are a number of ways we would like to advance this work. The biggest log jam is the use of open source, small models that can be run locally and without high API costs. The one proprietary LLM we used was [OpenAIs gpt-4-1106-preview model](https://platform.openai.com/docs/models), a state-of-the-art model that was used for evaluating the responses to the synthetically generated Q&A sets we created in our [evaluation notebook found here](./src/rag-eval.ipynb). With additional resources, we would want to use the highest quality LLMs for the tasks of embeddings, Q&A, synthetic Q&A generation, etc that are available as found here on the [Hugging Face Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).
#### Compute + Storage
Our intention is to productionalize this application. This will require cloud computing resources through a service such as AWS. For the purposes of this project, we decided to focus first on free-tier resources. Our plan is the following:
1. Use Amazon Redshift and/or DocumentDB services for vectorstores of embedded files.
2. Use AmazonS3 instances or HuggingFace Endpoint (Pro tier) to store high-parameter, high-performing LLMs.
3. Build AWS Glue scripts to manage databases and connect with UI for performing RAG.
## References
[1] https://tll.mit.edu/teaching-resources/course-design/syllabus/

[2] https://apnews.com/article/high-school-student-lawsuit-artificial-intelligence-8f1283b517b2ed95c2bac63f9c5cb0b9 

[3] https://huggingface.co/thenlper/gte-small 

[4] https://platform.openai.com/docs/models 

[5] https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard