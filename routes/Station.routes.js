const express = require('express');
const Authmiddleware = require('../middleware/Auth.middleware')
const { 
    getAllStations,
    createStation,
    updateStation,
    deleteStation,
    createBulkStations,
    getPrediction,
    getStationById
} = require('../controllers/Station.controller');

const router = express.Router();

// --- Station Data Routes ---
// GET /api/stations - Fetches all station data
router.get('/', getAllStations);
// POST /api/stations - Creates a new station
router.post('/', createStation);

// PUT /api/stations/:id - Updates a station by its ID
router.put('/:id', updateStation);

// DELETE /api/stations/:id - Deletes a station by its ID
router.delete('/:id', deleteStation);
// --- Prediction Routes ---
// POST /api/predict - Forwards request to the ML model

router.route('/bulk')
    .post(createBulkStations);

router.post('/predict', getPrediction);

router.get('/:id', getStationById);

module.exports = router;
