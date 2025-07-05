# F1 Race Predictor 🏎️

A machine learning-powered Streamlit web application that predicts Formula 1 race results using historical data, qualifying positions, and free practice results.

## 🚀 Features

- **Historical Data Analysis**: Analyzes past F1 race data, qualifying results, and practice sessions
- **Machine Learning Models**: Uses statistical modeling and ML algorithms to predict race outcomes
- **Interactive Web Interface**: Modern Streamlit application for easy interaction
- **Real-time Predictions**: Get predictions for upcoming races based on current season data
- **Driver Performance Analysis**: Track individual driver performance trends
- **Team Performance Insights**: Analyze team performance patterns
- **Interactive Visualizations**: Beautiful charts and graphs using Plotly

## 📊 Prediction Models

- **Qualifying to Race Correlation**: Analyzes how qualifying positions translate to race results
- **Practice Session Impact**: Evaluates the influence of free practice performance on race outcomes
- **Historical Pattern Recognition**: Identifies recurring patterns in driver and team performance
- **Weather and Track Conditions**: Considers environmental factors affecting race results

## 🛠️ Technology Stack

- **Web Framework**: Streamlit
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: Plotly, matplotlib, seaborn
- **Data Sources**: F1 API, historical datasets

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/F1Predictor.git
   cd F1Predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
F1Predictor/
├── streamlit_app.py        # Main Streamlit application
├── models/
│   ├── __init__.py
│   ├── data_processor.py   # Data preprocessing and feature engineering
│   ├── prediction_model.py # ML models for race predictions
│   └── evaluation.py       # Model evaluation and metrics
├── data/
│   ├── raw/               # Raw F1 data files
│   ├── processed/         # Processed datasets
│   └── scrapers/          # Data collection scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Usage

### Web Interface
1. Start the application: `streamlit run streamlit_app.py`
2. Navigate to the web interface at `http://localhost:8501`
3. Use the sidebar to navigate between different pages:
   - **🏠 Home**: Overview and quick demo
   - **🎯 Race Prediction**: Input race data and get predictions
   - **📊 Analysis**: Historical data analysis and visualizations
   - **📈 Model Performance**: Model accuracy and comparison
   - **ℹ️ About**: App information and features

### Features Available
- **Race Prediction**: Input qualifying positions, track, and weather conditions
- **Practice Data**: Optional practice session times for more accurate predictions
- **Interactive Charts**: Visualize predictions and historical trends
- **Driver Comparison**: Compare multiple drivers' performance
- **Track Analysis**: See how different tracks affect performance

## 📈 Model Performance

- **Accuracy**: ~75-80% for top 5 predictions
- **Features**: 50+ engineered features from historical data
- **Training Data**: 2010-2023 F1 seasons
- **Update Frequency**: Weekly during F1 season

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- F1 community for data and insights
- Open-source ML libraries
- F1 API providers
- Streamlit for the amazing web framework

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

**Disclaimer**: This application is for educational and entertainment purposes. Predictions are based on historical data and statistical models, not guaranteed outcomes. 