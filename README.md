# Fridge Chef – RAG-based Recipe Recommendation Engine

## Problem Description
Home cooks often struggle to decide what to make with the ingredients they already have in their fridge or pantry. Searching for recipes online can be time-consuming and may return results requiring unavailable ingredients. This leads to wasted time, food spoilage, and missed opportunities for creative cooking.

**Fridge Chef** addresses this problem by using a Retrieval-Augmented Generation (RAG) approach to recommend vegetarian and vegan recipes based on the user’s available ingredients. By narrowing the recipe suggestions to the ingredients at hand and preferred cuisines (Indian, Chinese, Italian, and Thai), Fridge Chef helps users:
- Reduce food waste by making use of available produce.
- Save time in meal planning.
- Explore diverse cuisines without guesswork.

## Dataset
The dataset was generated using ChatGPT and is stored in the `Data` folder as `RecipeData.json`.

### Data Structure
Each recipe in the dataset is represented as a JSON object with the following fields:

| Field                  | Description |
|------------------------|-------------|
| `dish_name`            | Name of the dish (e.g., "Chana Masala") |
| `cuisine`              | Cuisine type (Indian, Chinese, Italian, Thai) |
| `diet`                 | Dietary type ("Vegetarian" or "Vegan") |
| `tags`                 | Keywords related to the dish (e.g., "spicy", "quick") |
| `main_ingredients`     | Core ingredients used in the dish |
| `cooking_time_minutes` | Estimated preparation and cooking time in minutes |
| `difficulty`           | Difficulty level ("Easy", "Medium", "Hard") |
| `ingredients_full`     | Full list of ingredients and quantities |
| `instructions`         | Step-by-step cooking instructions |
| `substitutions`        | Suggested ingredient alternatives |
| `flavor_notes`         | Description of the taste and flavor profile |

Example record:
```json
{
  "dish_name": "Vegetable Pad Thai",
  "cuisine": "Thai",
  "diet": "Vegan",
  "tags": ["noodles", "quick", "stir-fry"],
  "main_ingredients": ["rice noodles", "tofu", "bean sprouts", "peanuts"],
  "cooking_time_minutes": 25,
  "difficulty": "Medium",
  "ingredients_full": ["200g rice noodles", "150g tofu", "100g bean sprouts", "2 tbsp soy sauce", "1 tbsp tamarind paste", "peanuts for garnish"],
  "instructions": "Soak noodles. Stir-fry tofu until golden. Add vegetables and sauce. Toss with noodles. Garnish with peanuts.",
  "substitutions": ["Use almond butter instead of peanuts for nut allergies"],
  "flavor_notes": "Savory with a balance of sweet and tangy"
}


## Evaluation
For the code for evaluating the system, you can check the [notebook/rag-test.ipynb](notebooks/rag-test.ipynb) notebook.

### Retrieval
The basic approach - using minsearch without using any boosting gives the following metrics:
* hit_rate: 97% [0.9755102040816327], 
* MRR: 83% [0.836089245221898]

The improved version (with better boosting):
* hit_rate: 97% [0.9755102040816327], 
* MRR: 90% [0.9050413022351798]

The best boosting parameters:
```python
boost = {
    'dish_name': 2.49,
    'cuisine': 2.16,
    'diet': 2.745,
    'tags': 0.23,
    'main_ingredients': 1.631,
    'cooking_time_minutes': 0.39,
    'difficulty': 2.64,
    'ingredients_full': 1.73
} 
```

### RAG

### Monitoring

### Ingestion