const Station = require('../models/Station.models');
const axios = require('axios');

/**
 * @desc    Get all station data
 * @route   GET /api/stations
 * @access  Public
 */
const getAllStations = async (req, res) => {
    try {
        // For demonstration, seed the database if it's empty
        const count = await Station.countDocuments();
        if (count === 0) {
            console.log('No stations found, seeding database with sample data...');
            await Station.create([
                { name: 'DWLR Bengaluru Urban', state: 'Karnataka', latitude: 12.9716, longitude: 77.5946, latestLevel: 15.2 },
                { name: 'DWLR Jaipur Rural', state: 'Rajasthan', latitude: 26.9124, longitude: 75.7873, latestLevel: 42.8 },
                { name: 'DWLR Pune Central', state: 'Maharashtra', latitude: 18.5204, longitude: 73.8567, latestLevel: 8.5 }
            ]);
        }
        const stations = await Station.find();
        res.json(stations);
    } catch (error) {
        res.status(500).json({ message: 'Server Error: ' + error.message });
    }
};

/**
 * @desc    Proxy request to ML model for prediction
 * @route   POST /api/predict
 * @access  Public
 */
const getPrediction = async (req, res) => {
    const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000/predict';

    try {
        const { features } = req.body;
        if (!features) {
            return res.status(400).json({ error: 'Features are required for prediction.' });
        }
        
        console.log(`Forwarding prediction request to: ${ML_API_URL}`);
        
        const mlResponse = await axios.post(ML_API_URL, { features });
        
        res.json(mlResponse.data);

    } catch (error) {
        console.error('Error calling ML model:', error.message);
        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({ error: 'The prediction service is currently unavailable.' });
        }
        res.status(500).json({ error: 'An internal error occurred while getting the prediction.' });
    }
};

module.exports = {
    getAllStations,
    getPrediction
};
