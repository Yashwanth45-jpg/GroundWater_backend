const mongoose = require('mongoose');
require('dotenv').config()

function ConnectToDb() {
    mongoose.connect(process.env.MONGO_URI).then(()=>{
        console.log('Connected to Db');
    })
    .catch(err=>{
        console.log(err);
    })
}

module.exports = ConnectToDb;