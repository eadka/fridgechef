import ingest
from openai import OpenAI
from dotenv import load_dotenv
import os
from time import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
index = ingest.load_index()

def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5
    )

    return results

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


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def rag(query,model='gpt-4o-mini'):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt,model=model)

    t1 = time()
    took = t1 - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": 0,
        "relevance_explanation": "RELEVANT",
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": len(answer.split()),
        "total_tokens": len(prompt.split())+len(answer.split()),
        "eval_prompt_tokens": 0,
        "eval_completion_tokens": 0,
        "eval_total_tokens": 0,
        "openai_cost": 0,
        # "relevance": relevance.get("Relevance", "UNKNOWN"),
        # "relevance_explanation": relevance.get(
        #     "Explanation", "Failed to parse evaluation"
        # ),
        # "prompt_tokens": token_stats["prompt_tokens"],
        # "completion_tokens": token_stats["completion_tokens"],
        # "total_tokens": token_stats["total_tokens"],
        # "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        # "eval_completion_tokens": rel_token_stats["completion_tokens"],
        # "eval_total_tokens": rel_token_stats["total_tokens"],
        # "openai_cost": openai_cost,
    }

    return answer_data

