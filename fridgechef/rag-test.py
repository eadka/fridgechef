#!/usr/bin/env python
# coding: utf-8

# ### Ingestion

# In[1]:


# Import the datafile
import requests
data_url = 'https://raw.githubusercontent.com/eadka/fridgechef/main/Data/RecipeData.json'
data_response = requests.get(data_url)
recipes_data = data_response.json()


# In[2]:


# Ensuring all the data has strings because minsearch, under the hood uses TfidfVectorizer and expects each text_field to be a string
for recipe in recipes_data:
    for field in ["dish_name",  "cuisine",  "diet", "tags",  "main_ingredients", 
                 "cooking_time_minutes", "difficulty",  "ingredients_full", 
                 "instructions", "substitutions", "flavor_notes"]:
        value = recipe.get(field, "")
        if isinstance(value,list):
            recipe[field] = " ".join(map(str,value)) # join the list into string
        elif not isinstance(value, str):
            recipe[field] = str(value) # convert numbers to string


# In[3]:


# Search engine and indexing
import minsearch

# Indexing the document
index = minsearch.Index(
    text_fields=["dish_name",  "cuisine",  "diet", "tags",  "main_ingredients", 
                 "cooking_time_minutes", "difficulty",  "ingredients_full", 
                 "instructions", "substitutions", "flavor_notes"],
    keyword_fields=[]
)


# In[4]:


index.fit(recipes_data)


# In[5]:


query = 'Give me recipes for carrots and beans'


# In[6]:


index.search(query,num_results=2)


# ### RAG Flow

# In[7]:


# Open AI for LLM integration
from openai import OpenAI

client = OpenAI()


# In[8]:


# response = client.chat.completions.create(
#     model='gpt-4o-mini',
#     messages=[{"role": "user", "content": query}]
# )

# response.choices[0].message.content


# In[9]:


# Defining the RAG flow
def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5
    )

    return results


# In[10]:


recipes_data[0]


# In[11]:


# prompt_template = """
# You're a "Fridge Chef", a helpful cooking assistant. 
# The user will give you a list of vegetables or ingredients they have available.
# Base your answer only on the recipes in the CONTEXT.
# If you cannot find an exact match, suggest the closest dishes using the available ingredients.

# When answering:
# - Include the dish name, cuisine, diet type, main ingredients, and cooking time.
# - Provide short cooking instructions based on the CONTEXT.
# - Suggest possible ingredient substitutions if given in the CONTEXT.
# - If multiple dishes fit, return the top 3–5 most relevant recipes.

# QUESTION: {question}

# CONTEXT: 
# {context}
# """.strip()

# entry_template = """
# dish_name: {dish_name}
# cuisine: {cuisine}
# diet: {diet}
# tags: {tags}
# main_ingredients: {main_ingredients}
# cooking_time_minutes: {cooking_time_minutes}
# difficulty: {difficulty}
# ingredients_full: {ingredients_full}
# instructions: {instructions}
# substitutions: {substitutions}
# flavor_notes: {flavor_notes}
# """.strip()

# def build_prompt(query, search_results):
#     context = ""

#     for doc in search_results:
#         context = context + entry_template.format(**doc) + "\n\n"

#     prompt = prompt_template.format(question=query, context=context).strip()
#     return prompt


# In[12]:


prompt_template = """
You're a "Fridge Chef", a helpful cooking assistant. 
The user will give you a list of vegetables or ingredients they have available.
Base your answer only on the recipes in the CONTEXT.
If you cannot find an exact match, suggest the closest dishes using the available ingredients.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

entry_template = """
dish_name: {dish_name}
cuisine: {cuisine}
diet: {diet}
tags: {tags}
main_ingredients: {main_ingredients}
cooking_time_minutes: {cooking_time_minutes}
difficulty: {difficulty}
ingredients_full: {ingredients_full}
instructions: {instructions}
substitutions: {substitutions}
flavor_notes: {flavor_notes}
""".strip()

def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[13]:


search_results = search(query)
prompt = build_prompt(query, search_results)


# In[14]:


print(prompt)


# In[76]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[78]:


def rag(query,model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt,model=model)
    return answer


# In[17]:


answer = rag('What is the main cooking technique used in the Vegetable Pad Thai?')
print(answer)


# ### Retrieval Evaluation

# In[18]:


import pandas as pd


# In[19]:


df_question = pd.read_csv('../Data/ground-truth-retrieval.csv')


# In[20]:


df_question.head()


# In[21]:


df_question.describe()


# In[22]:


ground_truth = df_question.to_dict(orient='records')


# In[23]:


ground_truth[0]


# In[24]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[25]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[26]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['dish_name'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[27]:


from tqdm.auto import tqdm


# In[28]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# ### Finding the best parameters

# In[29]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


# In[30]:


# Breaking the data into validation and test data sets
df_validation = df_question[:100]
df_test = df_question[100:]


# In[31]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Using float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)

        # Evaluate the objective function
        current_score = objective_function(current_params)

        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


# In[32]:


gt_val = df_validation.to_dict(orient='records')


# In[33]:


def minsearch_search(query,boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[34]:


# mrr
param_ranges = {
    'dish_name': (0.0,3.0),
    'cuisine': (0.0,3.0),
    'diet': (0.0,3.0),
    'tags': (0.0,3.0),
    'main_ingredients': (0.0,3.0),
    'cooking_time_minutes': (0.0,3.0),
    'difficulty': (0.0,3.0),
    'ingredients_full': (0.0,3.0)
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[35]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[36]:


# mrr
def minsearch_improved(query):
    boost = {'dish_name': 2.49,
        'cuisine': 2.16,
        'diet': 2.745,
        'tags': 0.23,
        'main_ingredients': 1.631,
        'cooking_time_minutes': 0.39,
        'difficulty': 2.64,
        'ingredients_full': 1.73
        } 

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# ### RAG evaluation

# #### gpt-4o-mini

# In[37]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[40]:


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


# In[42]:


print(question, answer_llm)


# In[45]:


prompt = prompt2_template.format(question=question, answer_llm = answer_llm)
print(prompt)


# In[46]:


llm(prompt)


# In[57]:


# evaluations = []

# for record in tqdm(ground_truth):
#     id = record['id']

#     if id in evaluations:
#         continue

#     question = record['question']
#     answer_llm = rag(question,model='gpt-4o-mini') 

#     prompt = prompt2_template.format(
#         question=question,
#         answer_llm=answer_llm
#     )

#     evaluation = llm(prompt)
#     evaluation = json.loads(evaluation)

#     evaluations.append((record, answer_llm, evaluation))


# In[58]:


evaluations


# In[61]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])
df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])


# In[63]:


del df_eval['record']
del df_eval['evaluation']


# In[65]:


df_eval


# In[67]:


df_eval.relevance.value_counts()


# In[74]:


df_eval[df_eval.relevance=='NON_RELEVANT']


# In[79]:


df_eval.relevance.value_counts(normalize=True)


# In[81]:


df_eval.to_csv('../Data/rag-eval-gpt-4o-mini.csv', index=False)


# #### gpt-3.5-turbo

# In[82]:


evaluations_gpt35turbo = []

for record in tqdm(ground_truth):
    id = record['id']

    if id in evaluations:
        continue

    question = record['question']
    answer_llm = rag(question, model ='gpt-3.5-turbo') 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations_gpt35turbo.append((record, answer_llm, evaluation))


# In[83]:


df_eval_35turbo = pd.DataFrame(evaluations_gpt35turbo, columns=['record', 'answer', 'evaluation'])

df_eval_35turbo['id'] = df_eval_35turbo.record.apply(lambda d: d['id'])
df_eval_35turbo['question'] = df_eval_35turbo.record.apply(lambda d: d['question'])
df_eval_35turbo['relevance'] = df_eval_35turbo.evaluation.apply(lambda d: d['Relevance'])
df_eval_35turbo['explanation'] = df_eval_35turbo.evaluation.apply(lambda d: d['Explanation'])


# In[84]:


df_eval_35turbo.relevance.value_counts()


# In[85]:


df_eval_35turbo.relevance.value_counts(normalize=True)


# In[ ]:


df_eval_35turbo.to_csv('../Data/rag-eval-gpt35turbo.csv', index=False)


# In[ ]:





# In[ ]:




