### Language Classifier ML App

This app was built using flask for the api and htmx for the backend.
The ML model is a nearest prototype classifier that analyses the frequency of trigrams (soon to be n-grams) in the text to be classified.

## How the model works
To identify the language a text is written in, the model matches the frequency of trigrams in the incoming text to the frequencies of trigrams in languages the model is trained on.
The trained language with the most similar trigram frequencies is inferred as the language the text is written in.


(Technical explanation of the model:)
The model represents trigram frequencies of trained languages in a inner product (vector) space (with trigrams as its basis vectors).
When a text is supplied to be classified, the model represents its trigram frequencies as new vector in this vector space and finds the 'nearest' vector to it with the euclidean inner product.
This closest vector is inferred as the language the text is written in.


## How the web app works:
### Flask (rest) API:
  - When the site is first accessed a get request is sent to the api which loads the models training data (trained_data.pkl).
  - As the user types input, a get request is sent at each character change to model.py which infers the language of the text that has been typed so far.
  - This is then sent to the front end to result.html to be rendered by htmx on the front end.
  - The simplicity of this app and flask's ability to seamlessly communicate with the model (model.py) made it the ideal choice for this web app's api over the benefits of other backend frameworks (including FastAPI or Express.js's speed, or Django's scalability). 

### HTMX front end:
  - HTMX is used to render the result of the classifiers inference. This was chosen to avoid rendering the entire index.html file every time the user types a character, resulting in a much smoother web app.
    Using HTMX is especially important as the app is hosted on render.com on the free web server, which has quite limited computation speeds.
  - HTMX is provided a target in index.html to render result.html once a result is recieved from the API.
  - The web server runs very smoothly when hosted locally, and only experiences a minor lag on render.com's server.
