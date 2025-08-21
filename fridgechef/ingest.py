# ### Ingestion

import requests
import minsearch


def load_index(data_path='https://raw.githubusercontent.com/eadka/fridgechef/main/Data/RecipeData.json'):

    data_response = requests.get(data_path)
    recipes_data = data_response.json()

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

    # Search engine and indexing
    # Indexing the document
    index = minsearch.Index(
        text_fields=[
            "dish_name",  
            "cuisine",  
            "diet", 
            "tags",  
            "main_ingredients", 
            "cooking_time_minutes", 
            "difficulty",  
            "ingredients_full", 
            "instructions", 
            "substitutions", 
            "flavor_notes"],
        keyword_fields=[]
    )

    index.fit(recipes_data)
    return index