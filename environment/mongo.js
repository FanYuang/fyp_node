const MongoClient = require('mongodb').MongoClient;

const mongoose = require('mongoose');
const conn = mongoose.createConnection(

    'mongodb://localhost:27017/fyp', {
    useNewUrlParser: true,
    useUnifiedTopology: true
  }
)

conn.on('open', () => {
  console.log('mongo connected!');
})
conn.on('err', (err) => {
  console.log('err:' + err);
})
conn.on('close', () => {
  console.log('mongo disconnected!');
})

let Timeschema=new mongoose.Schema({
    setuptime:Number,
    querytime:Number,
    method:String,
    distribution:String,
    num:Number,
    prop:Object
})

let Time=conn.model('Time', Timeschema);

module.exports = {
  conn: conn,
  Time:Time,

};