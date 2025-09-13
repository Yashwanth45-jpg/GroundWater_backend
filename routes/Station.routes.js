const express = require('express');
const Authmiddleware = require('../middleware/Auth.middleware')
const { 
    getAllStations, 
    getPrediction 
} = require('../controllers/station.controller');

const router = express.Router();

// --- Station Data Routes ---
// GET /api/stations - Fetches all station data
router.get('/', getAllStations);

// --- Prediction Routes ---
// POST /api/predict - Forwards request to the ML model
router.post('/predict', getPrediction);


module.exports = router;
