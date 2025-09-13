const mongoose = require('mongoose');

const stationSchema = new mongoose.Schema({
  name: { type: String, required: true },
  state: { type: String, required: true },
  latitude: { type: Number, required: true },
  longitude: { type: Number, required: true },
  latestLevel: { type: Number, required: true },
  lastUpdated: { type: Date, default: Date.now }
});

const Station = mongoose.model('Station', stationSchema);

module.exports = Station;
