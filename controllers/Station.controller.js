const Station = require('../models/Station.models');
const mongoose = require('mongoose')

const getAllStations = async (req, res) => {
    try {
        const stations = await Station.find();
        if (stations.length === 0) {
            return res.status(404).json({ message: 'No stations found.' });
        }
        res.json(stations);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

const getStationById = async (req, res) => {
    try {
        const { id } = req.params;
        if (!mongoose.Types.ObjectId.isValid(id)) {
            return res.status(400).json({ message: 'Invalid station ID format.' });
        }
        const station = await Station.findById(id);
        if (!station) {
            return res.status(404).json({ message: 'Station not found.' });
        }
        res.status(200).json(station);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching station: ' + error.message });
    }
};

/**
 * @desc    Create a new station
 * @route   POST /api/stations
 * @access  Private (requires auth)
 */
const createStation = async (req, res) => {
    const { name, state, latitude, longitude, latestLevel } = req.body;

    if (!name || !state || !latitude || !longitude || latestLevel === undefined) {
        return res.status(400).json({ message: 'Please provide all required fields.' });
    }

    try {
        const newStation = new Station({
            name,
            state,
            latitude,
            longitude,
            latestLevel
        });

        const savedStation = await newStation.save();
        res.status(201).json(savedStation);
    } catch (error) {
        res.status(500).json({ message: 'Error creating station.', error: error.message });
    }
};

// POST /api/stations/bulk
const createBulkStations = async (req, res) => {
    if (!Array.isArray(req.body) || req.body.length === 0) {
        return res.status(400).json({ message: 'Request body must be a non-empty array.' });
    }
    try {
        const createdStations = await Station.insertMany(req.body);
        res.status(201).json({ message: `Successfully inserted ${createdStations.length} stations.`, data: createdStations });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

/**
 * @desc    Update an existing station
 * @route   PUT /api/stations/:id
 * @access  Private (requires auth)
 */
const updateStation = async (req, res) => {
    try {
        const stationId = req.params.id;
        const updatedStation = await Station.findByIdAndUpdate(
            stationId, 
            req.body, 
            { new: true, runValidators: true } // {new: true} returns the updated document
        );

        if (!updatedStation) {
            return res.status(404).json({ message: 'Station not found.' });
        }

        res.json(updatedStation);
    } catch (error) {
        res.status(500).json({ message: 'Error updating station.', error: error.message });
    }
};

/**
 * @desc    Delete a station
 * @route   DELETE /api/stations/:id
 * @access  Private (requires auth)
 */
const deleteStation = async (req, res) => {
    try {
        const stationId = req.params.id;
        const deletedStation = await Station.findByIdAndDelete(stationId);

        if (!deletedStation) {
            return res.status(404).json({ message: 'Station not found.' });
        }

        res.json({ message: 'Station successfully deleted.', id: stationId });
    } catch (error) {
        res.status(500).json({ message: 'Error deleting station.', error: error.message });
    }
};

/**
 * @desc    Proxy request to ML model for prediction
 * @route   POST /api/predict
 * @access  Public
*/


const getPrediction = async (req, res) => {
    console.log('Received request for a prediction. Generating mock data...');

    try {
        // --- MOCK PREDICTION LOGIC ---
        // Instead of calling an external AI model, we'll generate random data.
        
        // 1. Generate a random number of days until a potential crisis.
        const days_until_crisis = Math.floor(Math.random() * 60) + 5; // Random number between 5 and 64

        // 2. Generate a random confidence score.
        const confidence_score = Math.random() * (0.98 - 0.75) + 0.75; // Random score between 75% and 98%

        // 3. Determine recommendations and message based on the crisis day.
        let crisis_message = '';
        let recommendations = [];

        if (days_until_crisis <= 15) {
            crisis_message = "Critical: Action required within the next 2 weeks.";
            recommendations = [
                "Implement immediate water conservation measures.",
                "Alert local water authorities for potential shortages.",
                "Consider using alternative water sources if available.",
                "Reduce all non-essential water usage immediately."
            ];
        } else if (days_until_crisis <= 30) {
            crisis_message = "Warning: Water levels are declining, monitor closely.";
            recommendations = [
                "Encourage voluntary water conservation in the community.",
                "Check for and repair any known leaks in the water system.",
                "Plan for potential seasonal shortages."
            ];
        } else {
            crisis_message = "Stable: Current water levels are adequate.";
            recommendations = [
                "Continue with regular monitoring practices.",
                "Maintain current water usage patterns.",
                "Promote sustainable long-term water management."
            ];
        }

        // 4. Assemble the final prediction object.
        const mockPrediction = {
            days_until_crisis,
            crisis_message,
            confidence_score,
            recommendations,
        };

        // 5. Send the mock prediction back to the frontend.
        res.status(200).json(mockPrediction);

    } catch (error) {
        console.error('Error generating mock prediction:', error.message);
        res.status(500).json({ message: 'An internal error occurred while generating the prediction.' });
    }
};

module.exports = {
    getAllStations,
    getStationById,
    createStation,
    createBulkStations,
    updateStation,
    deleteStation,
    getPrediction,
};