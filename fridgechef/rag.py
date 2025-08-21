import ingest
from openai import OpenAI

client = OpenAI()
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
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt,model=model)
    return answer

