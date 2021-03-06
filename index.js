const express = require('express')
const app = express();
const mongoose = require('mongoose');
const mongo = require('./environment/mongo');
const axios=require('axios');
const port = 4000
var gaussian = require('gaussian');
var AVLTree = require('@yetzt/binary-search-tree').AVLTree;
var sizeof = require('object-sizeof');
var dataset = [];
var testset=[];
let mean = 0;
let variance = 100000000;
let num_nor = 20000;
let low=0;
let high=10000000;
let num_uni = 10000000;
var distribution;
const tf = require('@tensorflow/tfjs');

app.get("/tfboy",(req,res)=>{
   
    res.send('Hello TensorFlow');
})

app.get('/', (req, res) => {
    res.send('Hello World!')
})


function generatenum_uniform(low, high, num_uni) {
    let arr = [];

    
    for (let i = 0; i < num_uni; i++) {

        let sample = parseInt(Math.random() * (high - low) + low);
        arr.push(sample);    
    }

    return arr;
}
function generatenum_normal(mean, variance, num_nor) {
    let arr = [];
    distribution = gaussian(mean, variance);
    for (let i = 0; i < num_nor; i++) {

        let sample = Math.round(distribution.ppf(Math.random()));
        arr.push(sample);
    }
    return arr;
}

app.get('/generatenum_normal', (req, res) => {

    dataset = generatenum_normal(mean, variance, num_nor);

    testset=generatenum_normal(mean, variance, num_nor);


    console.log(dataset);
    res.send('ok');
})

app.get('/hashtable_normal', (req, res) => {


   

    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let mapp = new Map();
    for (let i = 0; i < dataset.length; i++) {
        mapp.set(dataset[i], i);
    }
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
  
    let index = 0;

    //建index时间
    let prev=(new Date()).valueOf();
   
    let map = new Map();
    for (let i = 0; i < dataset.length; i++) {
        map.set(dataset[i], i);
    }

    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    
 
    prev=(new Date()).valueOf();
    while (index < num_nor) {
   
        if (map.has(testset[index])) {
            let result = map.get(testset[index]);
           
        }
        else {
            let result=-1;
        }
        index++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);

    let obj={};
    let prop={};
    prop.mean=mean;
    prop.variance=variance;
   
    obj.method="hashtable";
    obj.distribution="normal";
    obj.num=num_nor;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');
})

app.get('/binarysearch_normal', (req, res) => {

    let set=dataset.slice(0);
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = set.sort((a, b) => a - b).slice(0);;
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let sett=dataset.slice(0);
    let prev=(new Date()).valueOf();
    let array = sett.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);
    console.log(array);
    

   
    let index = 0;
    function binsearch(nums, target) {
        let low = 0, high = nums.length - 1;
        while (low <= high) {
            const mid = Math.floor((high - low) / 2) + low;
            const num = nums[mid];
            if (num === target) {
                return mid;
            } else if (num > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    };
    prev=(new Date()).valueOf();
    while (index < num_nor) {

        let result = binsearch(array, testset[index]);
        //console.log(result, index);
        index++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.mean=mean;
    prop.variance=variance;
   
    obj.method="binarysearch";
    obj.distribution="normal";
    obj.num=num_nor;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');
})

app.get("/avl_normal",(req,res)=>{
   

    let index = 0;
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let aavl=new AVLTree();
    for (let i = 0; i < dataset.length; i++) {
        aavl.insert(dataset[i], i);
    }

    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    
    let prev=(new Date()).valueOf();
    let avl=new AVLTree();
    for (let i = 0; i < dataset.length; i++) {
        avl.insert(dataset[i], i);
    }
    let next=(new Date()).valueOf();
    let setuptime=next-prev;

    console.log(prev,next,setuptime);
    prev=(new Date()).valueOf();
    while (index < num_nor) {

        if (avl.search(testset[index]).length>0)
        {
            let result = avl.search(testset[index]);
            //console.log(result);
        }

    else{
        let result=-1;
    }
    index++;
    }
    next=(new Date()).valueOf();
   index=0;
    while (index < num_nor) {

        if (avl.search(testset[index]).length>0)
        {
            let result;
            //console.log(result);
        }
  
    else{
        let result=-1;
    }
    index++;
    }
    let final=(new Date()).valueOf();
    let querytime=next-prev-final+next;
    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.mean=mean;
    prop.variance=variance;
   
    obj.method="avl";
    obj.distribution="normal";
    obj.num=num_nor;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');

})
app.get('/trick_normal', (req, res) => {


    let settt=dataset.slice(0);
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = settt.sort((a, b) => a - b).slice(0);
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    
    
    let i= 0;
    let sett=dataset.slice(0);
    let prev=(new Date()).valueOf();


    let array = sett.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);

    function tricksearch(set, index, target) {
        if (!set[index])
        {

        }

        else if  (set[index] < target){
            while (set[index]<target)
            {
                index++;

            }
            if (set[index]==target)
            return index;
            else
            return -1;

        }
        else if (set[index]>target)
        {
            while (set[index]<target)
            {
                index++;
            }
            if (set[index]==target)
            return index;
            else
            return -1;
        }
        


    }
    prev=(new Date()).valueOf();
    while (i < num_nor) {

        let result = tricksearch(array, Math.floor(num_nor * distribution.cdf(testset[i])), testset[i]);
        //console.log(result, i);
        i++;
    }
    next=(new Date()).valueOf();

    i=0;
    while (i < num_nor) {

        let result = Math.floor(num_nor * distribution.cdf(testset[i]));
  
        //console.log(result, i);
        i++;
    }
    let final=(new Date()).valueOf();
    let querytime=next-prev-final+next;
    console.log(prev,next,querytime,final);
    let obj={};
    let prop={};
    prop.mean=mean;
    prop.variance=variance;
   
    obj.method="trick";
    obj.distribution="normal";
    obj.num=num_nor;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send("ok");
})


app.get('/generatenum_uniform', (req, res) => {
  
    dataset = generatenum_uniform(low, high, num_uni);
    testset= generatenum_uniform(low, high, num_uni);



    console.log(dataset);
    res.send('ok');
})



//binary search uniform distribution
app.get('/binarysearch_uniform', (req, res) => {
    let set=dataset.slice(0);
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = set.sort((a, b) => a - b).slice(0);;
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let sett=dataset.slice(0);
    let prev=(new Date()).valueOf();
    let array = sett.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);
    
 


  
    let index = 0;
    function binsearch(nums, target) {
        let low = 0, high = nums.length - 1;
        while (low <= high) {
            const mid = Math.floor((high - low) / 2) + low;
            const num = nums[mid];
            if (num === target) {
                return mid;
            } else if (num > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    };
    prev=(new Date()).valueOf();
    while (index < num_uni) {

        let result = binsearch(array, testset[index]);
        //console.log(result, index);
        index++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.low=low;
    prop.high=high;
   
    obj.method="binarysearch";
    obj.distribution="uniform";
    obj.num=num_uni;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');
})

//trick method uniform distribution
app.get('/trick_uniform', (req, res) => {



    
    let set=dataset.slice(0);
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = set.sort((a, b) => a - b).slice(0);;
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let sett=dataset.slice(0);
    let prev=(new Date()).valueOf();
    let array = sett.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);

    let index = 0;
    function tricksearch(set, index, target) {
        if (!set[index])
        {

        }

        else if  (set[index] < target){
            while (set[index]<target)
            {
                index++;

            }
            if (set[index]==target)
            return index;
            else
            return -1;

        }
        else if (set[index]>target)
        {
            while (set[index]<target)
            {
                index++;
            }
            if (set[index]==target)
            return index;
            else
            return -1;
        }


    }
    prev=(new Date()).valueOf();
    while (index < num_uni) {

        let result = tricksearch(array, Math.floor(num_uni * (testset[index] - low) / (high - low)), testset[index]);
     
        index++;
    }
    next=(new Date()).valueOf();
    i=0;
    while (i < num_nor) {

        let result = Math.floor(num_uni * (testset[index] - low) / (high - low));
  
        //console.log(result, i);
        i++;
    }
    let final=(new Date()).valueOf();
    let querytime=next-prev-final+next;

    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.low=low;
    prop.high=high;
   
    obj.method="trick";
    obj.distribution="uniform";
    obj.num=num_uni;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send("ok");
})

//哈希表 uniform distribution
app.get('/hashtable_uniform', (req, res) => {


  


 
    let index = 0;
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let mapp = new Map();
    for (let i = 0; i < dataset.length; i++) {
        mapp.set(dataset[i], i);
    }
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
 
    console.log("mapp用了内存"+(usedafter-used)+"Mb");
    //建index时间
    let prev=(new Date()).valueOf();
    let map = new Map();
    for (let i = 0; i < dataset.length; i++) {
        map.set(dataset[i], i);
    }
    let next=(new Date()).valueOf();
    let setuptime=next-prev;

    console.log(prev,next,setuptime);
    prev=(new Date()).valueOf();
    while (index < num_uni) {

        if (map.has(testset[index])) {
            let result = map.get(testset[index]);
            //console.log(result);
        }
        else {
            let result=-1;
            //console.log("mou");
        }
        index++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.low=low;
    prop.high=high;
   
    obj.method="hashtable";
    obj.distribution="uniform";
    obj.num=num_uni;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');
})


app.get("/avl_uniform",(req,res)=>{

    let index = 0;
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let aavl=new AVLTree();
    for (let i = 0; i < dataset.length; i++) {
        aavl.insert(dataset[i], i);
    }

    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let prev=(new Date()).valueOf();
    let avl=new AVLTree();
    for (let i = 0; i < dataset.length; i++) {
        avl.insert(dataset[i], i);
    }
    let next=(new Date()).valueOf();
    let setuptime=next-prev;

    console.log(prev,next,setuptime);
    prev=(new Date()).valueOf();
    while (index < num_uni) {

        if (avl.search(testset[index]).length>0)
        {
            let result = avl.search(testset[index]);
        }
         
        else{
            let result=-1;
        }
    
        index++;
    }
    next=(new Date()).valueOf();
    index=0;
    while (index < num_uni) {

        if (avl.search(testset[index]).length>0)
        {
            let result;
            //console.log(result);
        }
        
        else{
            let result=-1;
        }
    
        index++;
    }
    let final=(new Date()).valueOf();
    let querytime=next-prev-final+next;
    console.log(prev,next,querytime);
    let obj={};
    let prop={};
    prop.low=low;
    prop.high=high;
   
    obj.method="avl";
    obj.distribution="uniform";
    obj.num=num_uni;
    obj.prop=prop;
    obj.querytime=querytime;
    obj.setuptime=setuptime;
    obj.size=usedafter-used;
    let data = new mongo.Result(obj);
    data.save();
    res.send('ok');

})

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})