from flask import Flask

# Create a Flask app
appp = Flask(__name__)

# Define a route and its corresponding function
@appp.route("/")
def home():
    return "Hello, Flask!"

# Run the app
if __name__ == "__main__":
    appp.run(debug=True)
