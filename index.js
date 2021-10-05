
const express = require('express')
const app = express()
const mongoose = require('mongoose');
const mongo = require('./environment/mongo');
const port = 3000
var gaussian = require('gaussian');
var AVLTree = require('@yetzt/binary-search-tree').AVLTree;
var sizeof = require('object-sizeof');
var dataset = [];
let mean = 0;
let variance = 1000000;
let num_nor = 2000;
let low=0;
let high=1000000;
let num_uni = 1000000;

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
    let distribution = gaussian(mean, variance);
    for (let i = 0; i < num_nor; i++) {

        let sample = Math.round(distribution.ppf(Math.random()));
        arr.push(sample);
    }
    return arr;
}

app.get('/generatenum_normal', (req, res) => {

    dataset = generatenum_normal(mean, variance, num_nor);



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
    let testset = generatenum_normal(mean, variance, num_nor);
    let index = 0;

    //建index时间
    let prev=(new Date()).valueOf();
   
    let map = new Map();
    for (let i = 0; i < dataset.length; i++) {
        map.set(dataset[i], i);
    }

    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    
    console.log(prev,next,setuptime,sizeof(map));
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
    let data = new mongo.Time(obj);
    data.save();
    res.send('ok');
})

app.get('/binarysearch_normal', (req, res) => {

    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = dataset.sort((a, b) => a - b);
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let prev=(new Date()).valueOf();
    let array = dataset.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);
    console.log(array);
    

    let testset = generatenum_normal(mean, variance, num_nor);
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
    res.send('ok');
})

app.get("/avl_normal",(req,res)=>{
   
    let testset = generatenum_normal(mean, variance, num_nor);
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

        if (avl.search(testset[index]).length>0) {
            let result = avl.search(testset[index]);
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
    res.send('ok');

})
app.get('/trick_normal', (req, res) => {


   
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = dataset.sort((a, b) => a - b);
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    
    let testset = generatenum_normal(mean, variance, num_nor);
    let i= 0;
    
    let prev=(new Date()).valueOf();


    let array = dataset.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);

    function tricksearch(set, index, target) {
        if (set[index] == target) {
            return index;
        }
        else if (index < 0) {
            return -1;
        }
        else if (index > set.length) {
            return -1;
        }
        else if (set[index] < target && set[index + 1] > target) {
            return -1;
        }
        else if (set[index] < target) {
            return tricksearch(set, index + 1, target)
        }
        else if (set[index] > target) {
            return tricksearch(set, index - 1, target)
        }


    }
    prev=(new Date()).valueOf();
    while (i < num_nor) {

        let result = tricksearch(array, Math.floor(num_nor * distribution.cdf(testset[i])), testset[i]);
        //console.log(result, i);
        i++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);
    res.send("ok");
})


app.get('/generatenum_uniform', (req, res) => {
  
    dataset = generatenum_uniform(low, high, num_uni);



    console.log(dataset);
    res.send('ok');
})



//binary search uniform distribution
app.get('/binarysearch_uniform', (req, res) => {
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = dataset.sort((a, b) => a - b);
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let prev=(new Date()).valueOf();
    let array = dataset.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);
    console.log(array);
 


    let testset = generatenum_uniform(low, high, num_uni);
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
    res.send('ok');
})

//trick method uniform distribution
app.get('/trick_uniform', (req, res) => {



    
   
    let used = process.memoryUsage().heapUsed / 1024 / 1024;
    let arr = dataset.sort((a, b) => a - b);
    let usedafter = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log("用了内存"+(usedafter-used)+"Mb");
    let prev=(new Date()).valueOf();
    let array = dataset.sort((a, b) => a - b);
    let next=(new Date()).valueOf();
    let setuptime=next-prev;
    console.log(prev,next,setuptime);
    let testset = generatenum_uniform(low, high, num_uni);
    let index = 0;
    function tricksearch(set, index, target) {
        if (set[index] == target) {
            return index;
        }
        else if (index < 0) {
            return -1;
        }
        else if (index > set.length) {
            return -1;
        }
        else if (set[index] < target && set[index + 1] > target) {
            return -1;
        }
        else if (set[index] < target) {
            return tricksearch(set, index + 1, target)
        }
        else if (set[index] > target) {
            return tricksearch(set, index - 1, target)
        }


    }
    prev=(new Date()).valueOf();
    while (index < num_uni) {

        let result = tricksearch(array, Math.floor(num_uni * (testset[index] - low) / (high - low)), testset[index]);
     
        index++;
    }
    next=(new Date()).valueOf();
    let querytime=next-prev;
    console.log(prev,next,querytime);

    res.send("ok");
})

//哈希表 uniform distribution
app.get('/hashtable_uniform', (req, res) => {


  


    let testset = generatenum_uniform(low, high, num_uni);
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

    res.send('ok');
})


app.get("/avl_uniform",(req,res)=>{
    let testset = generatenum_uniform(low, high, num_uni);
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

        if (avl.search(testset[index]).length>0) {
            let result = avl.search(testset[index]);
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
    res.send('ok');

})

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})