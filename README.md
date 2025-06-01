# Pixel Pusher - A Natural Language Game Recommender

[A web app](https://game-recommender-v2-156728017829.us-east4.run.app) to recommend games based on natural language input.

## Project Overview
Data has been fetched from the IGDB database using their API, narrowed down by the most import criteria. Data was organized, lightly cleaned, and then sent off to the cloud (RunPod GPU instance) to compute vector embeddings, this is for the user input to be compared against. User input is computed internally via a fintuned version of bge-small-en-v1.5 and then retrieved as vector embeddings in the same dimensions as the precomputed database. REST API construced with Flask for recommend, login, game save, and other functions. User data is stored via SQL tables. The backend code along with the data was Dockerized and then pushed to GCP for deployment. Frontend served from Flask app. Everything is self contained and within a single Docker image for simplicity.

## Tools Used:
- Python, HTML, JavaScript
  - pandas, numpy, sklearn
- Flask
- Google Cloud Platform
  - Cloud Run
  - Container Registry
- Docker
- Sentence Transformers
- Jupyter

## Results
API calls, backend and frontend connection, game recommendation retrieval, etc. are all successful. User can input what they want to play in natural language and get back game recommendations, the function retrieves the top 50 and then selects a random 10 (for a better feeling of randomness).

## Changes From V1
- Using a custom finetuned embedding model instead of the publicly available snowflake-arctic-embed2
- Added a random search
- Added info card for each game
- Changed website UI
- Changed deployment platform to GCP for simplicity

## Possible Next Steps
- Clean up data further, a lot of junk games present and it is unclear which games are fluff and which games people would actually play. Difficult to do so as data must be offloaded to a cloud GPU instance for embedding.
- Greater filter options. Only way to retrieve games is via natural langauge, hardcoded options may be valuable for greater filtering.

## Things to Note
- Accounts created may be deleted after leaving the website since SQL storage in GCR is ephemeral
- Game search may take longer than expected as resources allocated are minimal to keep costs as low as possible

![](https://github.com/razinlab/GameRecommenderV2/blob/478c433abdca21029fdb5defd6e0138e2b649733/Untitled%20video%20-%20Made%20with%20Clipchamp%20(3).gif)

