import modal
from modal import App, Image

# Setup

app = modal.App("hello") # defines a Modal app called “hello”
image = Image.debian_slim().pip_install("requests") # builds a lightweight container image (Debian slim) and preinstalls the requests library

# Hello!

# turns the following Python function into a Modal function (a remotely runnable, autoscaled serverless task) that runs inside the image you defined
# At runtime, it calls https://ipinfo.io/json to geolocate the public IP of the machine it’s running on (i.e., the Modal worker), not the caller’s laptop.
# It extracts city, region, and country from the JSON and returns a greeting string.
@app.function(image=image) 
def hello() -> str:
    import requests
    
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    city, region, country = data['city'], data['region'], data['country']
    return f"Hello from {city}, {region}, {country}!!"

# EU-region variant
@app.function(image=image, region="eu")
def hello_europe() -> str:
    import requests
    
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    city, region, country = data['city'], data['region'], data['country']
    return f"Hello from {city}, {region}, {country}!!"
