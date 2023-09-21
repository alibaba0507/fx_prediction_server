from flask import Flask, render_template, request, jsonify
from plot_generator import generate_plot,generate_supp_ress_plot  # Import the generate_plot function
from lstm_model_gen import generate_lstm_plot
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
    model_type= request.form.getlist('model-select')
    #print(shift)
    #print(loop)
    merged_list = []
    # Call the generate_plot function with user selections as parameters
    if model_type[0] == 'SGD' or model_type[0] == 'SGD_SR':
        plot_data = generate_plot(currency_pairs, periods,shift,loop)
        merged_list = plot_data
    if model_type[0] == 'SGD_SR':
        plot_lines = generate_supp_ress_plot(currency_pairs)
        merged_list = plot_lines + plot_data
    if model_type[0] == 'LSTM': 
        lstm_plot = generate_lstm_plot(currency_pairs, periods)
        merged_list = lstm_plot #plot_lines + plot_data
    
    # Return the plot data as JSON
    return jsonify(merged_list)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8000)