const express = require('express');
const cors = require('cors');
const stationRoutes = require('../routes/Station.routes');
const authRoutes = require('../routes/Auth.routes')
const cookieParser = require('cookie-parser')
const app = express();


app.use(cors());
app.use(express.json());
app.use(cookieParser())

app.use('/api/auth', authRoutes);

// --- API Routes ---
// All routes related to stations will be prefixed with /api
app.use('/api/stations', stationRoutes);

module.exports = app;