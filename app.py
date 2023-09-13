from flask import Flask, render_template, request, jsonify
from plot_generator import generate_plot,generate_supp_ress_plot  # Import the generate_plot function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot_route():
    currency_pairs = request.form.getlist('currency-pair')
    periods = request.form.getlist('period')
    shift = request.form.getlist('shift-backwards-period')
    loop = request.form.getlist('offset-times')
    #print(shift)
    #print(loop)
    # Call the generate_plot function with user selections as parameters
    plot_data = generate_plot(currency_pairs, periods,shift,loop)
    plot_lines = generate_supp_ress_plot(currency_pairs)
    merged_list = plot_lines + plot_data
    # Return the plot data as JSON
    return jsonify(merged_list)

if __name__ == '__main__':
    app.run(debug=True)
