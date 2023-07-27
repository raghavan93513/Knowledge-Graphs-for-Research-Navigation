# Research Paper Knowledge Graph and Recommender System

## Project Description

Knowledge graphs have emerged as powerful tools for representing and organizing information in a structured and interconnected manner. In this project, we aim to build a knowledge graph and a recommender system for research papers. 

This project is divided into three parts:

1. **Charting the Research Landscape**: Building a knowledge graph that represents the interconnectedness of a wide pool of research papers.
2. **The Search for Knowledge**: Designing a recommender system that suggests similar and important research papers.
3. **Unveiling the Trailblazers**: Conducting an impact analysis to identify the research works that have made the most significant impact on other papers .

## Getting Started

### Prerequisites

- Python: Python is likely the primary language for this project, given its wide range of libraries that support data analysis and machine learning. Should understand to use different tools and libraries like:-
    - json
    - rdflib
    - holoviews
    - networkx
    - pickle
    - transformers
    - torch
    - sklearn
    - scipy
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - datetime
    - collections
    - nltk
    - re
 - OpenAI API key: if exploration of GraphGPT is required.
 - Dataset: Access to the dataset (https://bit.ly/balkanid-ds-dataset-download) is necessary. Knowledge of how to handle JSONL files may also be necessary as indicated by the mention of new_research_papers.jsonl (https://bit.ly/balkanid-new-data-download).
 - Understanding of Recommender Systems: You should be familiar with the concept of recommender systems, information retrieval, and have a basic understanding of algorithms used in these systems such as pagerank, HITS or others.
 - Knowledge of Knowledge Graphs: A basic understanding of Knowledge Graphs, RDF Framework, and SPARQL language.

### Installation

```pip install -r requirements.txt``` <br/>
Run to install all of the Python modules and packages listed in your requirements.txt file.

## How to Use

1) Download dataset.jsonl and new_research_papers.jsonl files in the same directory
2) Run the Task1.ipynb file first to construct the knowledge graph and view results.
3) Run EDA.ipynb and view results.
4) Run Task2.ipynb and view results.
5) Run Task3.ipynb and view results.

## Exploratory Data Analysis (EDA)

I performed an exploratory data analysis to understand the structure and characteristics of our dataset. 

### EDA Approach

This file reads the knowledge graph from a file, analyzes it, and provides various insights such as:
1) Total Connections: Counts and prints the total number of connections between the nodes in the graph.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b12.png)
<br>
2) Citation Connections: Checks for specific citation relationships using a defined citation predicate. This gives an insight into how many citation connections exist in the research paper dataset.
   
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b13.png)
<br>
3) Missing Values: A SPARQL query is used to extract information about papers, such as the title, publication date, and abstract. This data is then used to create a pandas DataFrame. Checks for any missing values in the data.
   
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b1.png)
<br>
4) Publication Trend: Plots the number of papers published each year, which can reveal trends over time.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b2.png)
<br>
5) Abstract Analysis: The script also extracts and plots the length and word count of the abstracts. It also calculates the mean and median of the word count.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b4.png)
<br>
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b5.png)
<br>
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b14.png)
<br>
6) Discipline Analysis: The code finds the number of papers in each discipline and ranks the disciplines by the number of papers. This shows which disciplines have the most research.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b9.png)
<br>
7) Author Analysis: The code counts the number of papers for each author and shows the authors with the most papers. This can reveal the most prolific authors.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b8.png)
<br>
8) Citation Analysis: It checks the citation counts for each paper and identifies the most frequently cited papers.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b7.png)
<br>
9) Keyword Analysis: The script counts the number of times a given keyword appears in the titles, abstracts, and bodies of the papers. This can help identify the papers most relevant to a specific topic.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b6.png)
<br>
10) Common Words Analysis: The script counts the number of occurrences of each word in the titles and abstracts of the papers (excluding common stop words), and lists the most common words. This can reveal common themes or topics.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b10.png)
<br>
11) Citation Connection Check: It checks for the existence of citation connections for every node in the graph and prints the nodes that have citation connections.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/b11.png)
<br>
## Part 1: Charting the Research Landscape

### The knowledge graph construction

Structure of my Research Paper Knowledge Graph
- paper_id
- metadata
    - authors
    - title
    - update_date
- discipline
- abstract
    - text
- body_text
    - text
- bib_entries
    - bib_entry_raw
    - contained_arXiv_ids
        - id - if equal to any paper_id, it will draw a link between the two papers
        - text
- ref_entries
    - type
    - caption
        - table
        - figure

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/a.png)
<br/>

The above graph was plotted using GraphGPT, which is open source. GraphGPT converts unstructured natural language into a knowledge graph. You can even pass in the synopsis of your favorite movie, a passage from a confusing Wikipedia page, or transcript from a video to generate a graph visualization of entities and their relationships.
<br/>
Website: https://graphgpt.vercel.app/
<br/>

### Visualization Examples 
Multiple screenshots are taken for each query
<br/>
1) Query to provide valuable insights about a particular discipline in the research paper knowledge graph. Eg) “Statistics”

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c1.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c2.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c3.png)

2) Query to retrieve all papers written by a particular author. This can be beneficial for someone who wants to explore all papers published by a certain author. Eg) “Sansit Patnaik”

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c7.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c8.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c9.png)

3) Query to find all papers on a specific topic. This could be useful for a researcher looking to find all papers on a specific topic. Eg) “Quantum”.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c4.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c5.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c6.png)

4) Query to find all papers citing a specific paper. This could be helpful for a researcher trying to gauge the impact of a specific work. Eg) “1807.06209”.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c10.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c11.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c12.png)

5) Query to give insights into the papers that discuss a specific topic or concept. Eg) “ML”.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c13.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c14.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c15.png)

6) Query to get all citations in the knowledge graph

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c16.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c17.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c18.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/c19.png)

## Part 2: The Search for Knowledge

### Approach

Our task is to implement a recommender system that suggests 5 similar and important research papers that a researcher could cite for a new research paper that he/she plans to write
The steps taken by me in this project are as follows:-
1.	Data Representation
The input data is stored in the form of an RDF graph, where each paper is represented as a node and citations are represented as directed edges between these nodes.
2.	Data Extraction
Information about each paper - its title, discipline, abstract, and citation data, is extracted from the RDF graph. The citation data is represented as an adjacency matrix, where each row and column represents a paper and an entry is 1 if the corresponding paper cites the corresponding cited paper.
3.	Feature Generation
For each paper, a feature vector is generated by applying a pre-trained BERT model to its abstract. BERT is a transformer-based machine learning model designed to generate meaningful representations of text. These feature vectors, or embeddings, capture semantic information about the abstracts, such as the topics they discuss and the ways in which they discuss them.
4.	Similarity Calculation
The cosine similarity between each pair of papers is calculated based on their BERT embedding. The cosine similarity is a measure of the cosine of the angle between two vectors, which in this case provides a measure of the semantic similarity between two papers. Cosine similarity calculates the angle between these two vectors in the multi-dimensional space. It doesn't consider the magnitude (length) of the vectors; it only looks at the direction. The similarity score is a value between -1 and 1.
5.	Model Training: 
A logistic regression model is trained to predict whether one paper cites another based on their cosine similarity. Logistic regression is a binary classification model that outputs the probability that a given input belongs to a particular class, which in this case is the class of paper pairs where the first paper cites the second paper.
6.	Recommendation Generation: 
For a new paper, its similarity with each existing paper is calculated, and these similarities are fed into the logistic regression model to predict the probability of citation. The papers are then ranked by their predicted probabilities of citation, and the top papers are recommended.

### Output Screenshots
<br/>
Recommendation 1
'Enhanced Accuracy in Galactic Disc Action Estimates through Perturbed Distribution Functions' (Physics)

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d1.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d2.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d3.png)

<br/>
Recommendation 2
'A multimodal analysis of Parkinson's disease patients' (Statistics):

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d4.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d5.png)

<br/>
Recommendation 3
'LOGO2-BongradPlus' (Computer Science):

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d6.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d7.png)
![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d8.png)

<br/>

## Part 3: Unveiling the Trailblazers

### Approach

The algorithms used for impact analysis to identify the research works that have made the most significant impact on their peers are:-
1.	PageRank: <br/>
PageRank is a method used to measure the importance or influence of a paper in your research knowledge graph. 
It considers both the number of citations a paper receives and the importance of the papers that cite it. 
The advantage of using Pagerank is that it accounts for the quality and relevance of the papers citing a particular paper. 
In other words, if a highly influential paper cites another paper, it will give more weight to that citation than a less influential paper. 
This way, Pagerank helps identify papers that have a significant impact on the overall network of research papers, considering the connections and influence of each citation.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d9.png)

2.	HITS - Hub Score: <br/>
HITS stands for "Hyperlink-Induced Topic Search," and in this context, we focus on the Hub Score component. 
The Hub Score measures the importance of a paper based on its ability to cite other relevant and influential papers.
The advantage of using the Hub Score is that it highlights papers that act as excellent sources of information and references. 
A paper with a high Hub Score is like a central hub that connects to many other important papers, making it a valuable resource for researchers and indicating that it has had a considerable impact in the field.
I am not using Authority score in HITS score because
    - Authority Score is more relevant when you are interested in finding papers that are highly cited by other papers. 
    - While this information is undoubtedly valuable, it might not directly indicate the paper's centrality or its role as a hub within the research network. 
    - The Authority Score helps to identify papers that are considered authoritative or influential due to the number of citations they have received.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d10.png)

3.	Eigenvector Centrality: <br/>
It is another method to determine the importance of a paper in the research knowledge graph.
It calculates the centrality of a paper by considering not only the number of its citations but also the importance of the papers that cite it. 
The advantage of Eigenvector Centrality is that it gives more weight to citations from papers that are themselves highly central and influential. 
A citation from a paper that is well-connected and cited by many other important papers will contribute more to the centrality of the cited paper. 
This helps to identify papers that are not just cited frequently but are cited by other crucial papers, indicating their significant impact on the research community.

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d11.png)

In the question there was a line stating “Note: Importance of a research paper is measured not just by the number of citations that it receives but also by the quality of the citations. The quality of a citation is in turn determined by the number and quality of its own citations.” This is why I chose Eigenvector Centrality Score instead of Degree Centrality Score.

### Impactful Papers

Normalization is important to ensure:- <br/>
1.	Comparable Scale: Normalization brings all scores to a comparable scale. Without normalization, scores produced by different metrics might have different scales or ranges, making direct comparisons or aggregations (like your weighted average) inaccurate or misleading.
2.	Prevents Bias: Without normalization, one scoring method might dominate the final combined score simply because its raw scores are naturally higher, not because it's necessarily a better indicator of relevance or importance. By normalizing the scores, each method contributes equally to the final score, as intended by your 1/3 weightage.
3.	Improved Stability: Normalized metrics are generally more stable to small changes in the network, making your system more robust. For instance, adding a few nodes might significantly change raw PageRank scores but have a much smaller impact on the normalized scores.
4.	Interpretability: Normalized scores often have a clear interpretation, such as "proportion of total importance". This can make them easier to understand and work with.
5.	Faster Convergence: When used within iterative algorithms (as in the case of PageRank), normalized scores can help the algorithm converge more quickly.
<br/>
Considering the importance of normalization, I decided to normalize the PageRank scores, Hub scores, and Eigenvector Centrality with equal weightage in my recommender.
<br/>

![alt text](https://github.com/BalkanID-University/balkanid-fte-hiring-task-vit-vellore-2023-raghavan93513/blob/main/Screenshots/d12.png)

<br/>
The top 5 impactful papers are:- <br/>
1) 2009.00516 <br/>
2) 1605.02688 <br/>
3) 1011.0352 <br/>
4) 1412.6980 <br/>
5) quant-ph/9705052 <br/>
<br/>
<br/>
For more information look at the Main Document (pdf)
