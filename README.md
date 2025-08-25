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
```

### Technologoes

* [Minsearch](https://github.com/alexeygrigorev/minsearch) - For full -text search
* OpenAI as an LLM
* Flask as the API interface (see []() for more information on Flask)



### Installation
🚀 Setup Instructions

Docker is the easiest way to run this application. If you don't want to use Docker
and want to run this locally, then you need to manually prepare the environment and install
the dependencies as specified below:

Follow these steps to run the **Fridge Chef RAG** project locally.

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/fridgechef.git
cd fridgechef
```

#### 2. Install dependencies
We use Pipenv for environment and dependency management.

```bash
pipenv install pipenv
```

Installing the dependencies:

```bash
pipenv install --dev
```

#### 3. Set your OpenAI API Key
This project requires an OpenAI API key to run.
Create a .env file in the project root and add the following line:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```
⚠️ Important: Do not share or commit your .env file. It is ignored by .gitignore.


### Running with Docker

The easiest way to run this application is with docker:
```bash
docker-compose up
```

### 4. Run the app
To run the application locally, run the Flask application. 

```bash
cd fridgechef/
pipenv shell
export POSTGRES_HOST=localhost
python app.py
```

### Preparing the application
Before we can use the app, we need to initialize the database.
This can be done by running the [`db_prep.py`](fridgechef/db_prep.py) script:

```bash
cd fridgechef
pipenv shell
export POSTGRES_HOST=localhost
python db_prep.py
```

### Testing the app
```bash
URL=http://localhost:5000
QUESTION="What type of bread is used for the Bruschetta recipe?"
DATA='{
    "question": "'${QUESTION}'"
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    ${URL}/question
```

The answer will look like this:

```json
{
  "answer": "The type of bread used for the Bruschetta recipe is baguette slices.",
  "conversation_id": "7702887c-e8f3-4310-b47f-dc54a1ccb353",
  "question": "What type of bread is used for the Bruschetta recipe?"
}

Sending feedback:

```bash

ID="7702887c-e8f3-4310-b47f-dc54a1ccb353"

URL=http://localhost:5000

FEEDBACK_DATA="{
  \"conversation_id\": \"${ID}\",
  \"feedback\": 1
}"

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${FEEDBACK_DATA}" \
    ${URL}/feedback
```

After sending it, you will receive the acknowledgement:

```json
{
  "message": "Feedback received for conversation 7702887c-e8f3-4310-b47f-dc54a1ccb353: 1"
}
```
Alternatevely, you can use [test.py](test.py) for testing it:

```bash
cd notebooks/
pipenv run python test.py
```

### Misc
Running Jupyter notebook for experiments:

```bash
cd notebooks
pipenv run jupyter notebook
```

### Evaluation
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
We used the LLM-as-a-judge metric to evaluate the quality of our RAG flow
For gpt-4o-mini out of the 490 records, we had:

* 474 (96%) RELEVANT
* 13 (2.5%) PARTLY_RELEVANT
* 3 (1.5%) NON_RELEVANT

For gpt-3.5-turbo out of the 490 records, we had:

* 459 (93%) RELEVANT
* 29 (6%) PARTLY_RELEVANT
* 2 (1%) NON_RELEVANT

### Monitoring

### Ingestion
The ingestion script is in [fridgechef/ingest.py](fridgechef/ingest.py) and it is run on the startup of the app in [fridgechef/rag.py](fridgechef/rag.py)

### Flask as the API Interface  

In this project, **Flask** serves as a lightweight API layer for the Fridge Chef RAG engine.  
It exposes HTTP endpoints so that external applications can interact with the system:  

- `POST /ask` → takes ingredients and returns recipe suggestions  
- `POST /feedback` → records user feedback  

Using Flask makes the project:  
- **Accessible**: clients can call the API via HTTP  
- **Modular**: separates RAG logic from the interface  
- **Extendable**: easy to add new endpoints (e.g., personalization, analytics)  

In our case, we can send a question to `http://localhost:5000/question`

Flask acts as the bridge between the recipe database, RAG pipeline, and end users.