from flask import Flask, render_template, request, jsonify
from plot_generator import generate_plot  # Import the generate_plot function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot_route():
    currency_pairs = request.form.getlist('currency-pair')
    periods = request.form.getlist('period')
    #print(currency_pairs)
    #print(periods)
    # Call the generate_plot function with user selections as parameters
    plot_data = generate_plot(currency_pairs, periods)

    # Return the plot data as JSON
    return jsonify(plot_data)

if __name__ == '__main__':
    app.run(debug=True)
