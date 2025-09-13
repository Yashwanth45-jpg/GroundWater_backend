require('dotenv').config();
const app  = require('./src/app')
const ConnectToDb = require('./src/db/db')

console.log("server created")


ConnectToDb();

app.listen(3000, ()=>{
    console.log("Sever running")
})