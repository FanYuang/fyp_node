/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { KernelBackend } from './backends/backend';
import { Environment, setEnvironmentGlobal } from './environment';
import { getGlobalNamespace } from './global_util';
import { Add, Cast, Identity } from './kernel_names';
import { getGradient, getKernel, getKernelsForBackend } from './kernel_registry';
import { Profiler } from './profiler';
import { backpropagateGradients, getFilteredNodesXToY } from './tape';
import { setTensorTracker, Tensor, Variable } from './tensor';
import { getTensorsInContainer } from './tensor_util';
import * as util from './util';
import { bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape } from './util';
import * as log from './log';
function isRegisteredKernelInvocation(kernelInvocation) {
    return kernelInvocation.kernelName != null;
}
class EngineState {
    constructor() {
        // Public since optimizers will use it.
        this.registeredVariables = {};
        this.nextTapeNodeId = 0;
        this.numBytes = 0;
        this.numTensors = 0;
        this.numStringTensors = 0;
        this.numDataBuffers = 0;
        // Number of nested tf.grad() statements when computing higher-order
        // gradients. E.g. `1` for first-order gradients and `2` for second-order
        // gradients. Used to track if the tape should be removed after a backprop.
        this.gradientDepth = 0;
        // Number of nested kernel calls. When kernel depth is greater than 1, we turn
        // off the tape.
        this.kernelDepth = 0;
        this.scopeStack = [];
        /**
         * Keeps track of the number of data moves during a kernel execution. We
         * maintain a stack since kernels can call other kernels, recursively.
         */
        this.numDataMovesStack = [];
        this.nextScopeId = 0;
        this.tensorInfo = new WeakMap();
        this.profiling = false;
        this.activeProfile = {
            newBytes: 0,
            newTensors: 0,
            peakBytes: 0,
            kernels: [],
            result: null,
            get kernelNames() {
                return Array.from(new Set(this.kernels.map(k => k.name)));
            }
        };
    }
    dispose() {
        for (const variableName in this.registeredVariables) {
            this.registeredVariables[variableName].dispose();
        }
    }
}
export class Engine {
    constructor(ENV) {
        this.ENV = ENV;
        this.registry = {};
        this.registryFactory = {};
        this.pendingBackendInitId = 0;
        this.state = new EngineState();
    }
    async ready() {
        if (this.pendingBackendInit != null) {
            return this.pendingBackendInit.then(() => { });
        }
        if (this.backendInstance != null) {
            return;
        }
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const success = await this.initializeBackend(backendName).success;
            if (success) {
                await this.setBackend(backendName);
                return;
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    get backend() {
        if (this.pendingBackendInit != null) {
            throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make ` +
                `sure to await tf.ready() or await tf.setBackend() before calling ` +
                `other methods`);
        }
        if (this.backendInstance == null) {
            const { name, asyncInit } = this.initializeBackendsAndReturnBest();
            if (asyncInit) {
                throw new Error(`The highest priority backend '${name}' has not yet been ` +
                    `initialized. Make sure to await tf.ready() or ` +
                    `await tf.setBackend() before calling other methods`);
            }
            this.setBackend(name);
        }
        return this.backendInstance;
    }
    backendNames() {
        return Object.keys(this.registryFactory);
    }
    findBackend(backendName) {
        if (!(backendName in this.registry)) {
            // If the backend hasn't been initialized but we have a registry entry for
            // it, initialize it and return it.
            if (backendName in this.registryFactory) {
                const { asyncInit } = this.initializeBackend(backendName);
                if (asyncInit) {
                    // Backend is not ready yet.
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return this.registry[backendName];
    }
    findBackendFactory(backendName) {
        if (!(backendName in this.registryFactory)) {
            return null;
        }
        return this.registryFactory[backendName].factory;
    }
    registerBackend(backendName, factory, priority = 1) {
        if (backendName in this.registryFactory) {
            log.warn(`${backendName} backend was already registered. ` +
                `Reusing existing backend factory.`);
            return false;
        }
        this.registryFactory[backendName] = { factory, priority };
        return true;
    }
    async setBackend(backendName) {
        if (this.registryFactory[backendName] == null) {
            throw new Error(`Backend name '${backendName}' not found in registry`);
        }
        this.backendName = backendName;
        if (this.registry[backendName] == null) {
            this.backendInstance = null;
            const { success, asyncInit } = this.initializeBackend(backendName);
            const result = asyncInit ? await success : success;
            if (!result) {
                return false;
            }
        }
        this.backendInstance = this.registry[backendName];
        this.setupRegisteredKernels();
        // Reset the profiler.
        this.profiler = new Profiler(this.backendInstance);
        return true;
    }
    setupRegisteredKernels() {
        const kernels = getKernelsForBackend(this.backendName);
        kernels.forEach(kernel => {
            if (kernel.setupFunc != null) {
                kernel.setupFunc(this.backendInstance);
            }
        });
    }
    disposeRegisteredKernels(backendName) {
        const kernels = getKernelsForBackend(backendName);
        kernels.forEach(kernel => {
            if (kernel.disposeFunc != null) {
                kernel.disposeFunc(this.registry[backendName]);
            }
        });
    }
    /**
     * Initializes a backend by looking up the backend name in the factory
     * registry and calling the factory method. Returns a boolean representing
     * whether the initialization of the backend suceeded. Throws an error if
     * there is no backend in the factory registry.
     */
    initializeBackend(backendName) {
        const registryFactoryEntry = this.registryFactory[backendName];
        if (registryFactoryEntry == null) {
            throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
        }
        try {
            const backend = registryFactoryEntry.factory();
            /* Test if the factory returns a promise.
            Done in a more liberal way than
            previous 'Promise.resolve(backend)===backend'
            as we needed to account for custom Promise
            implementations (e.g. Angular) */
            if (backend && !(backend instanceof KernelBackend) &&
                typeof backend.then === 'function') {
                const promiseId = ++this.pendingBackendInitId;
                const success = backend
                    .then(backendInstance => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.registry[backendName] = backendInstance;
                    this.pendingBackendInit = null;
                    return true;
                })
                    .catch(err => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.pendingBackendInit = null;
                    log.warn(`Initialization of backend ${backendName} failed`);
                    log.warn(err.stack || err.message);
                    return false;
                });
                this.pendingBackendInit = success;
                return { success, asyncInit: true };
            }
            else {
                this.registry[backendName] = backend;
                return { success: true, asyncInit: false };
            }
        }
        catch (err) {
            log.warn(`Initialization of backend ${backendName} failed`);
            log.warn(err.stack || err.message);
            return { success: false, asyncInit: false };
        }
    }
    removeBackend(backendName) {
        if (!(backendName in this.registryFactory)) {
            throw new Error(`${backendName} backend not found in registry`);
        }
        if (this.backendName === backendName && this.pendingBackendInit != null) {
            // There is a pending promise of the backend we want to remove. Make it
            // obsolete.
            this.pendingBackendInitId++;
        }
        if (backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        delete this.registryFactory[backendName];
        // Unset the backend if it is active.
        if (this.backendName === backendName) {
            this.pendingBackendInit = null;
            this.backendName = null;
            this.backendInstance = null;
        }
    }
    getSortedBackends() {
        if (Object.keys(this.registryFactory).length === 0) {
            throw new Error('No backend found in registry.');
        }
        return Object.keys(this.registryFactory).sort((a, b) => {
            // Highest priority comes first.
            return this.registryFactory[b].priority -
                this.registryFactory[a].priority;
        });
    }
    initializeBackendsAndReturnBest() {
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const { success, asyncInit } = this.initializeBackend(backendName);
            if (asyncInit || success) {
                return { name: backendName, asyncInit };
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    moveData(backend, dataId) {
        const info = this.state.tensorInfo.get(dataId);
        const srcBackend = info.backend;
        const values = this.readSync(dataId);
        const refCount = srcBackend.refCount(dataId);
        // Delete the tensor from the old backend and move it to the new
        // backend.
        srcBackend.disposeData(dataId, true);
        info.backend = backend;
        backend.move(dataId, values, info.shape, info.dtype, refCount);
        if (this.shouldCheckForMemLeaks()) {
            // Track the number of moves during a kernel execution to correctly
            // detect memory leaks.
            this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
        }
    }
    tidy(nameOrFn, fn) {
        let name = null;
        if (fn == null) {
            // Called with only 1 argument.
            if (typeof nameOrFn !== 'function') {
                throw new Error('Please provide a function to tidy()');
            }
            fn = nameOrFn;
        }
        else {
            // Called with 2 arguments.
            if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to tidy() must be a string');
            }
            if (typeof fn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to tidy() must be a function');
            }
            name = nameOrFn;
            // TODO(nsthorat,smilkov): Do operation logging and performance
            // profiling.
        }
        let result;
        return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
            result = fn();
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            return result;
        });
    }
    scopedRun(start, end, f) {
        start();
        try {
            const res = f();
            end();
            return res;
        }
        catch (ex) {
            end();
            throw ex;
        }
    }
    nextTensorId() {
        return Engine.nextTensorId++;
    }
    nextVariableId() {
        return Engine.nextVariableId++;
    }
    /**
     * This method is called instead of the public-facing tensor.clone() when
     * saving a tensor for backwards pass. It makes sure to add the clone
     * operation to the tape regardless of being called inside a kernel
     * execution.
     */
    clone(x) {
        const y = ENGINE.runKernel(Identity, { x });
        const inputs = { x };
        const grad = (dy) => ({
            x: () => {
                const dtype = 'float32';
                const gradInputs = { x: dy };
                const attrs = { dtype };
                return ENGINE.runKernel(Cast, gradInputs, 
                // tslint:disable-next-line: no-unnecessary-type-assertion
                attrs);
            }
        });
        const saved = [];
        this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
        return y;
    }
    /**
     * Execute a kernel with the given name and return the output tensor.
     *
     * @param kernelName The name of the kernel to execute.
     * @param inputs A map of input names to tensors.
     * @param attrs A map of attribute names to their values. An attribute is a
     *     primitive (non-tensor) input to the kernel.
     * @param inputsToSave A list of tensors, inputs to save for the backprop
     *     computation.
     * @param outputsToSave A list of booleans, specifying which output to save
     *     for the backprop computation. These are booleans since the output
     * tensors are not visible to the user.
     */
    runKernel(kernelName, inputs, attrs) {
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        const hasKernel = getKernel(kernelName, this.backendName) != null;
        if (!hasKernel) {
            throw new Error(`Kernel '${kernelName}' not registered for backend '${this.backendName}'`);
        }
        return this.runKernelFunc({ kernelName, inputs, attrs });
    }
    shouldCheckForMemLeaks() {
        return this.ENV.getBool('IS_TEST');
    }
    checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
        const numDataIdsAfter = this.backend.numDataIds();
        // Count the number of data ids associated with the result of the kernel.
        let numOutputDataIds = 0;
        outInfos.forEach(info => {
            // Complex numbers allocate 3 data ids, one for 'real', one for
            // 'imaginary', and one for the container that holds the former two.
            numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
        });
        // Account for the number of moves during kernel execution. A "data move"
        // can happen in the middle of a kernel execution, placing a new (key,value)
        // pair in the data storage. Since data moves have net zero effect (we
        // always remove the data from the old backend), we have to cancel them out
        // when detecting memory leaks.
        const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
        const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
        if (dataIdsLeaked > 0) {
            throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
        }
    }
    /**
     * Internal helper method to execute a kernel Func
     *
     * Use `runKernel` to execute kernels from outside of engine.
     */
    runKernelFunc(kernelParams) {
        let outputs;
        let saved = [];
        const isTapeOn = this.isTapeOn();
        const startingBytecount = this.state.numBytes;
        const startingNumTensors = this.state.numTensors;
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack.push(0);
        }
        let kernelFunc;
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        let out;
        const kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
            kernelParams.kernelName :
            this.state.activeScope != null ? this.state.activeScope.name : '';
        // Create the kernelFunc from either a registered kernel OR passed in
        // forward/backward functions (used by custom grad). In this context a
        // kernelFunc wraps a kernel implementation with some bookkeeping.
        if (isRegisteredKernelInvocation(kernelParams)) {
            const { kernelName, inputs, attrs } = kernelParams;
            if (this.backendName == null) {
                // backend has not been initialized yet (backend initialization is lazy
                // can be deferred until an op/ kernel is run).
                // The below getter has side effects that will try to initialize the
                // backend and set properties like this.backendName
                // tslint:disable-next-line: no-unused-expression
                this.backend;
            }
            const kernel = getKernel(kernelName, this.backendName);
            util.assert(kernel != null, () => `Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'`);
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                const outInfos = Array.isArray(out) ? out : [out];
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                }
                const outTensors = outInfos.map((outInfo) => {
                    // todo (yassogba) remove this option (Tensor) when node backend
                    // methods have been modularized and they all return tensorInfo.
                    // TensorInfos do not have a rank attribute.
                    if (outInfo.rank != null) {
                        return outInfo;
                    }
                    const { dataId, shape, dtype } = outInfo;
                    return this.makeTensorFromDataId(dataId, shape, dtype);
                });
                // Save any required inputs and outputs.
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since there would be no backprop for these tensors
                // (which would otherwise dispose them).
                if (isTapeOn) {
                    const tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                    saved = this.saveTensorsForBackwardMode(tensorsToSave);
                }
                return outTensors;
            };
        }
        else {
            const { forwardFunc } = kernelParams;
            // Running a customGrad op.
            const saveFunc = (tensors) => {
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since we would never run backprop, which disposes
                // the kept tensors.
                if (!isTapeOn) {
                    return;
                }
                saved = tensors.map(tensor => this.keep(this.clone(tensor)));
            };
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                const outs = (Array.isArray(out) ? out : [out]);
                if (this.shouldCheckForMemLeaks()) {
                    // Scope name is used to print a more helpful error message if needed.
                    this.checkKernelForMemLeak(kernelOrScopeName, numDataIdsBefore, outs);
                }
                return outs;
            };
        }
        //
        // Run the kernelFunc. Optionally profiling it.
        //
        const { inputs, attrs } = kernelParams;
        const backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
            null :
            kernelParams.backwardsFunc;
        let kernelProfile;
        this.scopedRun(
        // Stop recording to a tape when running a kernel.
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
            if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
                outputs = kernelFunc();
            }
            else {
                kernelProfile = this.profiler.profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
                if (this.ENV.getBool('DEBUG')) {
                    this.profiler.logKernelProfile(kernelProfile);
                }
                outputs = kernelProfile.outputs;
            }
        });
        if (isTapeOn) {
            this.addTapeNode(kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
        }
        if (this.state.profiling) {
            this.state.activeProfile.kernels.push({
                name: kernelOrScopeName,
                bytesAdded: this.state.numBytes - startingBytecount,
                totalBytesSnapshot: this.state.numBytes,
                tensorsAdded: this.state.numTensors - startingNumTensors,
                totalTensorsSnapshot: this.state.numTensors,
                inputShapes: Object.keys(inputs).map(key => inputs[key] != null ? inputs[key].shape : null),
                outputShapes: outputs.map(item => item.shape),
                kernelTimeMs: kernelProfile.timeMs,
                extraInfo: kernelProfile.extraInfo
            });
        }
        return (Array.isArray(out) ? outputs : outputs[0]);
    }
    /**
     * Saves tensors used in forward mode for use in backward mode.
     *
     * @param tensors the list of tensors to save.
     */
    saveTensorsForBackwardMode(tensors) {
        const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
        return saved;
    }
    /**
     * Returns a list of tensors to save for a given gradient calculation.
     *
     * @param kernelName name of kernel to look up gradient for.
     * @param inputs a map of input tensors.
     * @param outputs an array of output tensors from forward mode of kernel.
     */
    getTensorsForGradient(kernelName, inputs, outputs) {
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            const inputsToSave = gradConfig.inputsToSave || [];
            const outputsToSave = gradConfig.outputsToSave || [];
            // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
            // specified in inputsToSave will be saved.
            let inputTensorsToSave;
            if (gradConfig.saveAllInputs) {
                util.assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
            }
            else {
                inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
            }
            const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
            return inputTensorsToSave.concat(outputTensorsToSave);
        }
        // We return an empty list rather than throw an error because the kernel we
        // are looking up may not actually be relevant to backproping through the
        // overall function
        //
        // See 'does not error if irrelevant (pruned) ops are missing grads' test
        // in gradients_test.ts for an example.
        return [];
    }
    /**
     * Internal method used by public APIs for tensor creation. Makes a new
     * tensor with the provided shape, dtype and values. It always
     * creates a new data id and writes the values to the underlying backend.
     */
    makeTensor(values, shape, dtype, backend) {
        if (values == null) {
            throw new Error('Values passed to engine.makeTensor() are null');
        }
        dtype = dtype || 'float32';
        backend = backend || this.backend;
        let backendVals = values;
        if (dtype === 'string' && util.isString(values[0])) {
            backendVals = values.map(d => util.encodeString(d));
        }
        const dataId = backend.write(backendVals, shape, dtype);
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        // Count bytes for string tensors.
        if (dtype === 'string') {
            const info = this.state.tensorInfo.get(dataId);
            const newBytes = bytesFromStringArray(backendVals);
            this.state.numBytes += newBytes - info.bytes;
            info.bytes = newBytes;
        }
        return t;
    }
    /**
     * Internal method used by backends. Makes a new tensor
     * that is a wrapper around an existing data id. It doesn't create
     * a new data id, only increments the ref count used in memory tracking.
     */
    makeTensorFromDataId(dataId, shape, dtype, backend) {
        dtype = dtype || 'float32';
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        return t;
    }
    makeVariable(initialValue, trainable = true, name, dtype) {
        name = name || this.nextVariableId().toString();
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.cast(dtype);
        }
        const v = new Variable(initialValue, trainable, name, this.nextTensorId());
        if (this.state.registeredVariables[v.name] != null) {
            throw new Error(`Variable with name ${v.name} was already registered`);
        }
        this.state.registeredVariables[v.name] = v;
        this.incRef(v, this.backend);
        return v;
    }
    trackTensor(a, backend) {
        this.state.numTensors++;
        if (a.dtype === 'string') {
            this.state.numStringTensors++;
        }
        // Bytes for complex numbers are counted by their components. Bytes for
        // string tensors are counted when writing values.
        let bytes = 0;
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            bytes = a.size * util.bytesPerElement(a.dtype);
        }
        this.state.numBytes += bytes;
        if (!this.state.tensorInfo.has(a.dataId)) {
            this.state.numDataBuffers++;
            this.state.tensorInfo.set(a.dataId, {
                backend: backend || this.backend,
                dtype: a.dtype,
                shape: a.shape,
                bytes
            });
        }
        if (!(a instanceof Variable)) {
            this.track(a);
        }
    }
    // Track the tensor by dataId and increase the refCount for the dataId in the
    // backend.
    // TODO(pyu10055): This is currently used by makeVariable method, to increase
    // refCount on the backend for the dataId. It can potentially be replaced with
    // Identity op indead of calling backend directly.
    incRef(a, backend) {
        this.trackTensor(a, backend);
        this.backend.incRef(a.dataId);
    }
    removeDataId(dataId, backend) {
        if (this.state.tensorInfo.has(dataId) &&
            this.state.tensorInfo.get(dataId).backend === backend) {
            this.state.tensorInfo.delete(dataId);
            this.state.numDataBuffers--;
        }
    }
    disposeTensor(a) {
        if (!this.state.tensorInfo.has(a.dataId)) {
            return;
        }
        const info = this.state.tensorInfo.get(a.dataId);
        this.state.numTensors--;
        if (a.dtype === 'string') {
            this.state.numStringTensors--;
            this.state.numBytes -= info.bytes;
        }
        // Don't count bytes for complex numbers as they are counted by their
        // components.
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            const bytes = a.size * util.bytesPerElement(a.dtype);
            this.state.numBytes -= bytes;
        }
        // Remove the reference to dataId if backend dispose the data successfully
        if (info.backend.disposeData(a.dataId)) {
            this.removeDataId(a.dataId, info.backend);
        }
        // TODO(nsthorat): Construct an error and save the stack trace for
        // debugging when in debug mode. Creating a stack trace is too expensive
        // to do unconditionally.
    }
    disposeVariables() {
        for (const varName in this.state.registeredVariables) {
            const v = this.state.registeredVariables[varName];
            this.disposeVariable(v);
        }
    }
    disposeVariable(v) {
        this.disposeTensor(v);
        if (this.state.registeredVariables[v.name] != null) {
            delete this.state.registeredVariables[v.name];
        }
    }
    memory() {
        const info = this.backend.memory();
        info.numTensors = this.state.numTensors;
        info.numDataBuffers = this.state.numDataBuffers;
        info.numBytes = this.state.numBytes;
        if (this.state.numStringTensors > 0) {
            info.unreliable = true;
            if (info.reasons == null) {
                info.reasons = [];
            }
            info.reasons.push('Memory usage by string tensors is approximate ' +
                '(2 bytes per character)');
        }
        return info;
    }
    async profile(query) {
        this.state.profiling = true;
        const startBytes = this.state.numBytes;
        const startNumTensors = this.state.numTensors;
        this.state.activeProfile.kernels = [];
        this.state.activeProfile.result = await query();
        this.state.profiling = false;
        this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
        this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
        this.state.activeProfile.newTensors =
            this.state.numTensors - startNumTensors;
        for (const kernel of this.state.activeProfile.kernels) {
            kernel.kernelTimeMs = await kernel.kernelTimeMs;
            kernel.extraInfo = await kernel.extraInfo;
        }
        return this.state.activeProfile;
    }
    isTapeOn() {
        return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
    }
    addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
        const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            gradientsFunc = gradConfig.gradFunc;
        }
        if (gradientsFunc != null) {
            tapeNode.gradient = (dys) => {
                // TODO(smilkov): To optimize back-prop, pass dys that are not used in
                // the backprop graph to the user as null instead of zeros
                dys = dys.map((dy, i) => {
                    if (dy == null) {
                        const output = outputs[i];
                        const vals = util.makeZerosTypedArray(output.size, output.dtype);
                        return this.makeTensor(vals, output.shape, output.dtype);
                    }
                    return dy;
                });
                // Grad functions of ops with single outputs expect a dy, while ops
                // with multiple outputs expect dys (array of dy).
                return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
            };
        }
        this.state.activeTape.push(tapeNode);
    }
    keep(result) {
        result.kept = true;
        return result;
    }
    startTape() {
        if (this.state.gradientDepth === 0) {
            this.state.activeTape = [];
        }
        this.state.gradientDepth++;
    }
    endTape() {
        this.state.gradientDepth--;
    }
    /**
     * Start a scope. Use this with endScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    startScope(name) {
        const scopeInfo = {
            track: [],
            name: 'unnamed scope',
            id: this.state.nextScopeId++
        };
        if (name) {
            scopeInfo.name = name;
        }
        this.state.scopeStack.push(scopeInfo);
        this.state.activeScope = scopeInfo;
    }
    /**
     * End a scope. Use this with startScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    endScope(result) {
        const tensorsToTrackInParent = getTensorsInContainer(result);
        const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
        // Dispose the arrays tracked in this scope.
        for (let i = 0; i < this.state.activeScope.track.length; i++) {
            const tensor = this.state.activeScope.track[i];
            if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                tensor.dispose();
            }
        }
        const oldScope = this.state.scopeStack.pop();
        this.state.activeScope = this.state.scopeStack.length === 0 ?
            null :
            this.state.scopeStack[this.state.scopeStack.length - 1];
        // Track the current result in the parent scope.
        tensorsToTrackInParent.forEach(tensor => {
            // Only track the tensor if was allocated in the inner scope and is not
            // globally kept.
            if (!tensor.kept && tensor.scopeId === oldScope.id) {
                this.track(tensor);
            }
        });
    }
    /**
     * Returns gradients of `f` with respect to each of the `xs`. The gradients
     * returned are of the same length as `xs`, but some might be null if `f`
     * was not a function of that `x`. It also takes optional dy to multiply the
     * gradient, which defaults to `1`.
     */
    gradients(f, xs, dy, allowNoGradients = false) {
        util.assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
        if (dy != null && dy.dtype !== 'float32') {
            throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
        }
        const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
        util.assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
        // Filter out the nodes that don't connect x => y.
        const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
        if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
            throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                'that the f you passed encloses all operations that lead from x ' +
                'to y.');
        }
        return this.tidy('backward', () => {
            const accumulatedGradientMap = {};
            accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
            // Backprop gradients through the filtered nodes.
            backpropagateGradients(accumulatedGradientMap, filteredTape, 
            // Pass the tidy function to avoid circular dep with `tape.ts`.
            f => this.tidy(f), 
            // Pass an add function to avoide a circular dep with `tape.ts`.
            add);
            const grads = xs.map(x => accumulatedGradientMap[x.id]);
            if (this.state.gradientDepth === 0) {
                // This means that we are not computing higher-order gradients
                // and can clean up the tape.
                this.state.activeTape.forEach(node => {
                    for (const tensor of node.saved) {
                        tensor.dispose();
                    }
                });
                this.state.activeTape = null;
            }
            return { value: y, grads };
        });
    }
    customGrad(f) {
        util.assert(util.isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
        return (...inputs) => {
            util.assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                'tensors');
            let res;
            const inputMap = {};
            inputs.forEach((input, i) => {
                inputMap[i] = input;
            });
            const forwardFunc = (_, save) => {
                res = f(...[...inputs, save]);
                util.assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.value` is a tensor');
                util.assert(util.isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function.');
                return res.value;
            };
            const backwardsFunc = (dy, saved) => {
                const gradRes = res.gradFunc(dy, saved);
                const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                util.assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
                util.assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'a list of only tensors.');
                const gradMap = {};
                grads.forEach((grad, i) => {
                    gradMap[i] = () => grad;
                });
                return gradMap;
            };
            return this.runKernelFunc({
                forwardFunc,
                backwardsFunc,
                inputs: inputMap,
            });
        };
    }
    readSync(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readSync(dataId);
    }
    read(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.read(dataId);
    }
    async time(query) {
        const start = now();
        const timingInfo = await this.backend.time(query);
        timingInfo.wallMs = now() - start;
        return timingInfo;
    }
    /**
     * Tracks a Tensor in the current scope to be automatically cleaned up
     * when the current scope ends, and returns the value.
     *
     * @param result The Tensor to track in the current scope.
     */
    track(result) {
        if (this.state.activeScope != null) {
            result.scopeId = this.state.activeScope.id;
            this.state.activeScope.track.push(result);
        }
        return result;
    }
    get registeredVariables() {
        return this.state.registeredVariables;
    }
    /**
     * Resets the engine state. Removes all backends but does not remove
     * registered backend factories.
     */
    reset() {
        // Make any pending promise obsolete.
        this.pendingBackendInitId++;
        this.state.dispose();
        this.ENV.reset();
        this.state = new EngineState();
        for (const backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        this.backendName = null;
        this.backendInstance = null;
        this.pendingBackendInit = null;
    }
}
Engine.nextTensorId = 0;
Engine.nextVariableId = 0;
function ones(shape) {
    const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
    return ENGINE.makeTensor(values, shape, 'float32');
}
export function getOrMakeEngine() {
    const ns = getGlobalNamespace();
    if (ns._tfengine == null) {
        const environment = new Environment(ns);
        ns._tfengine = new Engine(environment);
    }
    setEnvironmentGlobal(ns._tfengine.ENV);
    // Tell the current tensor interface that the global engine is responsible
    // for tracking.
    setTensorTracker(() => ns._tfengine);
    return ns._tfengine;
}
export const ENGINE = getOrMakeEngine();
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
export function add(a, b) {
    // We duplicate Add here to avoid a circular dependency with add.ts.
    const inputs = { a, b };
    return ENGINE.runKernel(Add, inputs);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5naW5lLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9lbmdpbmUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUErQixhQUFhLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUMvRSxPQUFPLEVBQUMsV0FBVyxFQUFFLG9CQUFvQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQ2hFLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUNqRCxPQUFPLEVBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNuRCxPQUFPLEVBQUMsV0FBVyxFQUFFLFNBQVMsRUFBRSxvQkFBb0IsRUFBcUMsTUFBTSxtQkFBbUIsQ0FBQztBQUNuSCxPQUFPLEVBQWdCLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNuRCxPQUFPLEVBQUMsc0JBQXNCLEVBQUUsb0JBQW9CLEVBQVcsTUFBTSxRQUFRLENBQUM7QUFDOUUsT0FBTyxFQUFTLGdCQUFnQixFQUFFLE1BQU0sRUFBaUIsUUFBUSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBRW5GLE9BQU8sRUFBQyxxQkFBcUIsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVwRCxPQUFPLEtBQUssSUFBSSxNQUFNLFFBQVEsQ0FBQztBQUMvQixPQUFPLEVBQUMsb0JBQW9CLEVBQUUsa0JBQWtCLEVBQUUsR0FBRyxFQUFFLGFBQWEsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUNwRixPQUFPLEtBQUssR0FBRyxNQUFNLE9BQU8sQ0FBQztBQXNFN0IsU0FBUyw0QkFBNEIsQ0FFakMsZ0JBQ2dDO0lBRWxDLE9BQVEsZ0JBQWtELENBQUMsVUFBVSxJQUFJLElBQUksQ0FBQztBQUNoRixDQUFDO0FBRUQsTUFBTSxXQUFXO0lBQWpCO1FBQ0UsdUNBQXVDO1FBQ3ZDLHdCQUFtQixH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QUksTUFBTSxDQUFDLFdBQVcsQ0FBQyxDQUFDO0tBQ3hDO0lBQ0Qsb0JBQW9CLENBQUMsRUFBRSxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUV2QywwRUFBMEU7SUFDMUUsZ0JBQWdCO0lBQ2hCLGdCQUFnQixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUNyQyxPQUFPLEVBQUUsQ0FBQyxTQUFTLENBQUM7QUFDdEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxlQUFlLEVBQUUsQ0FBQztBQUV4Qzs7Ozs7R0FLRztBQUNILE1BQU0sVUFBVSxHQUFHLENBQUMsQ0FBUyxFQUFFLENBQVM7SUFDdEMsb0VBQW9FO0lBQ3BFLE1BQU0sTUFBTSxHQUFHLEVBQUMsQ0FBQyxFQUFFLENBQUMsRUFBQyxDQUFDO0lBQ3RCLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxHQUFHLEVBQUUsTUFBOEIsQ0FBQyxDQUFDO0FBQy9ELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7QmFja2VuZFRpbWluZ0luZm8sIERhdGFNb3ZlciwgS2VybmVsQmFja2VuZH0gZnJvbSAnLi9iYWNrZW5kcy9iYWNrZW5kJztcbmltcG9ydCB7RW52aXJvbm1lbnQsIHNldEVudmlyb25tZW50R2xvYmFsfSBmcm9tICcuL2Vudmlyb25tZW50JztcbmltcG9ydCB7Z2V0R2xvYmFsTmFtZXNwYWNlfSBmcm9tICcuL2dsb2JhbF91dGlsJztcbmltcG9ydCB7QWRkLCBDYXN0LCBJZGVudGl0eX0gZnJvbSAnLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtnZXRHcmFkaWVudCwgZ2V0S2VybmVsLCBnZXRLZXJuZWxzRm9yQmFja2VuZCwgR3JhZEZ1bmMsIE5hbWVkQXR0ck1hcCwgVGVuc29ySW5mb30gZnJvbSAnLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtLZXJuZWxQcm9maWxlLCBQcm9maWxlcn0gZnJvbSAnLi9wcm9maWxlcic7XG5pbXBvcnQge2JhY2twcm9wYWdhdGVHcmFkaWVudHMsIGdldEZpbHRlcmVkTm9kZXNYVG9ZLCBUYXBlTm9kZX0gZnJvbSAnLi90YXBlJztcbmltcG9ydCB7RGF0YUlkLCBzZXRUZW5zb3JUcmFja2VyLCBUZW5zb3IsIFRlbnNvclRyYWNrZXIsIFZhcmlhYmxlfSBmcm9tICcuL3RlbnNvcic7XG5pbXBvcnQge0dyYWRTYXZlRnVuYywgTmFtZWRUZW5zb3JNYXAsIE5hbWVkVmFyaWFibGVNYXAsIFRlbnNvckNvbnRhaW5lcn0gZnJvbSAnLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtnZXRUZW5zb3JzSW5Db250YWluZXJ9IGZyb20gJy4vdGVuc29yX3V0aWwnO1xuaW1wb3J0IHtCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZSwgRGF0YVZhbHVlc30gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4vdXRpbCc7XG5pbXBvcnQge2J5dGVzRnJvbVN0cmluZ0FycmF5LCBtYWtlT25lc1R5cGVkQXJyYXksIG5vdywgc2l6ZUZyb21TaGFwZX0gZnJvbSAnLi91dGlsJztcbmltcG9ydCAqIGFzIGxvZyBmcm9tICcuL2xvZyc7XG4vKipcbiAqIEEgZnVuY3Rpb24gdGhhdCBjb21wdXRlcyBhbiBvdXRwdXQuIFRoZSBzYXZlIGZ1bmN0aW9uIGlzIGZvciBzYXZpbmcgdGVuc29yc1xuICogY29tcHV0ZWQgaW4gdGhlIGZvcndhcmQgcGFzcywgdGhhdCB3ZSBuZWVkIGluIHRoZSBiYWNrd2FyZCBwYXNzLlxuICovXG5leHBvcnQgdHlwZSBGb3J3YXJkRnVuYzxUPiA9IChiYWNrZW5kOiBLZXJuZWxCYWNrZW5kLCBzYXZlPzogR3JhZFNhdmVGdW5jKSA9PiBUO1xuXG4vKipcbiAqIEBkb2NhbGlhcyAoYTogVGVuc29yLCBiOiBUZW5zb3IsLi4uLCBzYXZlPzogRnVuY3Rpb24pID0+IHtcbiAqICAgdmFsdWU6IFRlbnNvcixcbiAqICAgZ3JhZEZ1bmM6IChkeTogVGVuc29yLCBzYXZlZD86IE5hbWVkVGVuc29yTWFwKSA9PiBUZW5zb3IgfCBUZW5zb3JbXVxuICogfVxuICovXG5leHBvcnQgdHlwZSBDdXN0b21HcmFkaWVudEZ1bmM8VCBleHRlbmRzIFRlbnNvcj4gPVxuICAgICguLi5pbnB1dHM6IEFycmF5PFRlbnNvcnxHcmFkU2F2ZUZ1bmM+KSA9PiB7XG4gICAgICB2YWx1ZTogVDtcbiAgICAgIGdyYWRGdW5jOiAoZHk6IFQsIHNhdmVkOiBUZW5zb3JbXSkgPT4gVGVuc29yIHwgVGVuc29yW107XG4gICAgfTtcblxuZXhwb3J0IHR5cGUgTWVtb3J5SW5mbyA9IHtcbiAgbnVtVGVuc29yczogbnVtYmVyOyBudW1EYXRhQnVmZmVyczogbnVtYmVyOyBudW1CeXRlczogbnVtYmVyO1xuICB1bnJlbGlhYmxlPzogYm9vbGVhbjsgcmVhc29uczogc3RyaW5nW107XG59O1xuXG50eXBlIEtlcm5lbEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZzsgYnl0ZXNBZGRlZDogbnVtYmVyOyB0b3RhbEJ5dGVzU25hcHNob3Q6IG51bWJlcjtcbiAgdGVuc29yc0FkZGVkOiBudW1iZXI7XG4gIHRvdGFsVGVuc29yc1NuYXBzaG90OiBudW1iZXI7XG4gIGlucHV0U2hhcGVzOiBudW1iZXJbXVtdO1xuICBvdXRwdXRTaGFwZXM6IG51bWJlcltdW107XG4gIGtlcm5lbFRpbWVNczogbnVtYmVyIHwge2Vycm9yOiBzdHJpbmd9IHwgUHJvbWlzZTxudW1iZXJ8e2Vycm9yOiBzdHJpbmd9PjtcbiAgZXh0cmFJbmZvOiBzdHJpbmcgfCBQcm9taXNlPHN0cmluZz47XG59O1xuXG5leHBvcnQgdHlwZSBQcm9maWxlSW5mbyA9IHtcbiAgbmV3Qnl0ZXM6IG51bWJlcjsgbmV3VGVuc29yczogbnVtYmVyOyBwZWFrQnl0ZXM6IG51bWJlcjtcbiAga2VybmVsczogS2VybmVsSW5mb1tdO1xuICByZXN1bHQ6IFRlbnNvckNvbnRhaW5lcjtcbiAga2VybmVsTmFtZXM6IHN0cmluZ1tdO1xufTtcblxuZXhwb3J0IGludGVyZmFjZSBUaW1pbmdJbmZvIGV4dGVuZHMgQmFja2VuZFRpbWluZ0luZm8ge1xuICB3YWxsTXM6IG51bWJlcjtcbn1cblxuLyoqIEBkb2NhbGlhcyBGdW5jdGlvbiAqL1xuZXhwb3J0IHR5cGUgU2NvcGVGbjxUIGV4dGVuZHMgVGVuc29yQ29udGFpbmVyPiA9ICgpID0+IFQ7XG5cbmludGVyZmFjZSBTY29wZVN0YXRlIHtcbiAgdHJhY2s6IFRlbnNvcltdO1xuICBuYW1lOiBzdHJpbmc7XG4gIGlkOiBudW1iZXI7XG59XG5cbmludGVyZmFjZSBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJIGV4dGVuZHMgTmFtZWRUZW5zb3JNYXA+IHtcbiAga2VybmVsTmFtZTogc3RyaW5nO1xuICBpbnB1dHM6IEk7XG4gIGF0dHJzPzogTmFtZWRBdHRyTWFwO1xufVxuXG5pbnRlcmZhY2UgQ3VzdG9tR3JhZEtlcm5lbEludm9jYXRpb248VCBleHRlbmRzIFRlbnNvcnxUZW5zb3JbXSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgSSBleHRlbmRzIE5hbWVkVGVuc29yTWFwPiB7XG4gIGZvcndhcmRGdW5jOiBGb3J3YXJkRnVuYzxUPjtcbiAgYmFja3dhcmRzRnVuYzogKGR5OiBULCBzYXZlZDogVGVuc29yW10pID0+IHtcbiAgICBbUCBpbiBrZXlvZiBJXTogKCkgPT4gSVtQXVxuICB9O1xuICBpbnB1dHM6IEk7XG4gIGF0dHJzPzogTmFtZWRBdHRyTWFwO1xufVxuXG5mdW5jdGlvbiBpc1JlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPFQgZXh0ZW5kcyBUZW5zb3J8VGVuc29yW10sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJIGV4dGVuZHMgTmFtZWRUZW5zb3JNYXA+KFxuICAgIGtlcm5lbEludm9jYXRpb246IFJlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPEk+fFxuICAgIEN1c3RvbUdyYWRLZXJuZWxJbnZvY2F0aW9uPFQsIEk+KTpcbiAgICBrZXJuZWxJbnZvY2F0aW9uIGlzIFJlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPEk+IHtcbiAgcmV0dXJuIChrZXJuZWxJbnZvY2F0aW9uIGFzIFJlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPEk+KS5rZXJuZWxOYW1lICE9IG51bGw7XG59XG5cbmNsYXNzIEVuZ2luZVN0YXRlIHtcbiAgLy8gUHVibGljIHNpbmNlIG9wdGltaXplcnMgd2lsbCB1c2UgaXQuXG4gIHJlZ2lzdGVyZWRWYXJpYWJsZXM6IE5hbWVkVmFyaWFibGVNYXAgPSB7fTtcblxuICBuZXh0VGFwZU5vZGVJZCA9IDA7XG4gIG51bUJ5dGVzID0gMDtcbiAgbnVtVGVuc29ycyA9IDA7XG4gIG51bVN0cmluZ1RlbnNvcnMgPSAwO1xuICBudW1EYXRhQnVmZmVycyA9IDA7XG5cbiAgYWN0aXZlVGFwZTogVGFwZU5vZGVbXTtcbiAgLy8gTnVtYmVyIG9mIG5lc3RlZCB0Zi5ncmFkKCkgc3RhdGVtZW50cyB3aGVuIGNvbXB1dGluZyBoaWdoZXItb3JkZXJcbiAgLy8gZ3JhZGllbnRzLiBFLmcuIGAxYCBmb3IgZmlyc3Qtb3JkZXIgZ3JhZGllbnRzIGFuZCBgMmAgZm9yIHNlY29uZC1vcmRlclxuICAvLyBncmFkaWVudHMuIFVzZWQgdG8gdHJhY2sgaWYgdGhlIHRhcGUgc2hvdWxkIGJlIHJlbW92ZWQgYWZ0ZXIgYSBiYWNrcHJvcC5cbiAgZ3JhZGllbnREZXB0aCA9IDA7XG4gIC8vIE51bWJlciBvZiBuZXN0ZWQga2VybmVsIGNhbGxzLiBXaGVuIGtlcm5lbCBkZXB0aCBpcyBncmVhdGVyIHRoYW4gMSwgd2UgdHVyblxuICAvLyBvZmYgdGhlIHRhcGUuXG4gIGtlcm5lbERlcHRoID0gMDtcblxuICAvLyBLZWVwIFRlbnNvcnMgdGhhdCBwYXJhbGxlbCB0aGUgdGFwZXMuXG4gIGFjdGl2ZVNjb3BlOiBTY29wZVN0YXRlO1xuICBzY29wZVN0YWNrOiBTY29wZVN0YXRlW10gPSBbXTtcbiAgLyoqXG4gICAqIEtlZXBzIHRyYWNrIG9mIHRoZSBudW1iZXIgb2YgZGF0YSBtb3ZlcyBkdXJpbmcgYSBrZXJuZWwgZXhlY3V0aW9uLiBXZVxuICAgKiBtYWludGFpbiBhIHN0YWNrIHNpbmNlIGtlcm5lbHMgY2FuIGNhbGwgb3RoZXIga2VybmVscywgcmVjdXJzaXZlbHkuXG4gICAqL1xuICBudW1EYXRhTW92ZXNTdGFjazogbnVtYmVyW10gPSBbXTtcbiAgbmV4dFNjb3BlSWQgPSAwO1xuXG4gIHRlbnNvckluZm8gPSBuZXcgV2Vha01hcDxEYXRhSWQsIHtcbiAgICBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kLFxuICAgIGJ5dGVzOiBudW1iZXIsXG4gICAgZHR5cGU6IERhdGFUeXBlLFxuICAgIHNoYXBlOiBudW1iZXJbXVxuICB9PigpO1xuXG4gIHByb2ZpbGluZyA9IGZhbHNlO1xuICBhY3RpdmVQcm9maWxlOiBQcm9maWxlSW5mbyA9IHtcbiAgICBuZXdCeXRlczogMCxcbiAgICBuZXdUZW5zb3JzOiAwLFxuICAgIHBlYWtCeXRlczogMCxcbiAgICBrZXJuZWxzOiBbXSxcbiAgICByZXN1bHQ6IG51bGwsXG4gICAgZ2V0IGtlcm5lbE5hbWVzKCk6XG4gICAgICAgIHN0cmluZ1tdIHtcbiAgICAgICAgICByZXR1cm4gQXJyYXkuZnJvbShuZXcgU2V0KHRoaXMua2VybmVscy5tYXAoayA9PiBrLm5hbWUpKSk7XG4gICAgICAgIH1cbiAgfTtcblxuICBkaXNwb3NlKCkge1xuICAgIGZvciAoY29uc3QgdmFyaWFibGVOYW1lIGluIHRoaXMucmVnaXN0ZXJlZFZhcmlhYmxlcykge1xuICAgICAgdGhpcy5yZWdpc3RlcmVkVmFyaWFibGVzW3ZhcmlhYmxlTmFtZV0uZGlzcG9zZSgpO1xuICAgIH1cbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgRW5naW5lIGltcGxlbWVudHMgVGVuc29yVHJhY2tlciwgRGF0YU1vdmVyIHtcbiAgc3RhdGU6IEVuZ2luZVN0YXRlO1xuICBiYWNrZW5kTmFtZTogc3RyaW5nO1xuICByZWdpc3RyeToge1tpZDogc3RyaW5nXTogS2VybmVsQmFja2VuZH0gPSB7fTtcbiAgcmVnaXN0cnlGYWN0b3J5OiB7XG4gICAgW2lkOiBzdHJpbmddOiB7XG4gICAgICBmYWN0b3J5OiAoKSA9PiBLZXJuZWxCYWNrZW5kIHwgUHJvbWlzZTxLZXJuZWxCYWNrZW5kPixcbiAgICAgIHByaW9yaXR5OiBudW1iZXJcbiAgICB9XG4gIH0gPSB7fTtcblxuICBwcml2YXRlIHByb2ZpbGVyOiBQcm9maWxlcjtcbiAgcHJpdmF0ZSBiYWNrZW5kSW5zdGFuY2U6IEtlcm5lbEJhY2tlbmQ7XG4gIHByaXZhdGUgcGVuZGluZ0JhY2tlbmRJbml0OiBQcm9taXNlPGJvb2xlYW4+O1xuICBwcml2YXRlIHBlbmRpbmdCYWNrZW5kSW5pdElkID0gMDtcblxuICBjb25zdHJ1Y3RvcihwdWJsaWMgRU5WOiBFbnZpcm9ubWVudCkge1xuICAgIHRoaXMuc3RhdGUgPSBuZXcgRW5naW5lU3RhdGUoKTtcbiAgfVxuXG4gIGFzeW5jIHJlYWR5KCk6IFByb21pc2U8dm9pZD4ge1xuICAgIGlmICh0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCAhPSBudWxsKSB7XG4gICAgICByZXR1cm4gdGhpcy5wZW5kaW5nQmFja2VuZEluaXQudGhlbigoKSA9PiB7fSk7XG4gICAgfVxuICAgIGlmICh0aGlzLmJhY2tlbmRJbnN0YW5jZSAhPSBudWxsKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IHNvcnRlZEJhY2tlbmRzID0gdGhpcy5nZXRTb3J0ZWRCYWNrZW5kcygpO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzb3J0ZWRCYWNrZW5kcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgYmFja2VuZE5hbWUgPSBzb3J0ZWRCYWNrZW5kc1tpXTtcbiAgICAgIGNvbnN0IHN1Y2Nlc3MgPSBhd2FpdCB0aGlzLmluaXRpYWxpemVCYWNrZW5kKGJhY2tlbmROYW1lKS5zdWNjZXNzO1xuICAgICAgaWYgKHN1Y2Nlc3MpIHtcbiAgICAgICAgYXdhaXQgdGhpcy5zZXRCYWNrZW5kKGJhY2tlbmROYW1lKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYENvdWxkIG5vdCBpbml0aWFsaXplIGFueSBiYWNrZW5kcywgYWxsIGJhY2tlbmQgaW5pdGlhbGl6YXRpb25zIGAgK1xuICAgICAgICBgZmFpbGVkLmApO1xuICB9XG5cbiAgZ2V0IGJhY2tlbmQoKTogS2VybmVsQmFja2VuZCB7XG4gICAgaWYgKHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQmFja2VuZCAnJHt0aGlzLmJhY2tlbmROYW1lfScgaGFzIG5vdCB5ZXQgYmVlbiBpbml0aWFsaXplZC4gTWFrZSBgICtcbiAgICAgICAgICBgc3VyZSB0byBhd2FpdCB0Zi5yZWFkeSgpIG9yIGF3YWl0IHRmLnNldEJhY2tlbmQoKSBiZWZvcmUgY2FsbGluZyBgICtcbiAgICAgICAgICBgb3RoZXIgbWV0aG9kc2ApO1xuICAgIH1cbiAgICBpZiAodGhpcy5iYWNrZW5kSW5zdGFuY2UgPT0gbnVsbCkge1xuICAgICAgY29uc3Qge25hbWUsIGFzeW5jSW5pdH0gPSB0aGlzLmluaXRpYWxpemVCYWNrZW5kc0FuZFJldHVybkJlc3QoKTtcbiAgICAgIGlmIChhc3luY0luaXQpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgYFRoZSBoaWdoZXN0IHByaW9yaXR5IGJhY2tlbmQgJyR7bmFtZX0nIGhhcyBub3QgeWV0IGJlZW4gYCArXG4gICAgICAgICAgICBgaW5pdGlhbGl6ZWQuIE1ha2Ugc3VyZSB0byBhd2FpdCB0Zi5yZWFkeSgpIG9yIGAgK1xuICAgICAgICAgICAgYGF3YWl0IHRmLnNldEJhY2tlbmQoKSBiZWZvcmUgY2FsbGluZyBvdGhlciBtZXRob2RzYCk7XG4gICAgICB9XG4gICAgICB0aGlzLnNldEJhY2tlbmQobmFtZSk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmJhY2tlbmRJbnN0YW5jZTtcbiAgfVxuXG4gIGJhY2tlbmROYW1lcygpOiBzdHJpbmdbXSB7XG4gICAgcmV0dXJuIE9iamVjdC5rZXlzKHRoaXMucmVnaXN0cnlGYWN0b3J5KTtcbiAgfVxuXG4gIGZpbmRCYWNrZW5kKGJhY2tlbmROYW1lOiBzdHJpbmcpOiBLZXJuZWxCYWNrZW5kIHtcbiAgICBpZiAoIShiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5KSkge1xuICAgICAgLy8gSWYgdGhlIGJhY2tlbmQgaGFzbid0IGJlZW4gaW5pdGlhbGl6ZWQgYnV0IHdlIGhhdmUgYSByZWdpc3RyeSBlbnRyeSBmb3JcbiAgICAgIC8vIGl0LCBpbml0aWFsaXplIGl0IGFuZCByZXR1cm4gaXQuXG4gICAgICBpZiAoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeUZhY3RvcnkpIHtcbiAgICAgICAgY29uc3Qge2FzeW5jSW5pdH0gPSB0aGlzLmluaXRpYWxpemVCYWNrZW5kKGJhY2tlbmROYW1lKTtcbiAgICAgICAgaWYgKGFzeW5jSW5pdCkge1xuICAgICAgICAgIC8vIEJhY2tlbmQgaXMgbm90IHJlYWR5IHlldC5cbiAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXTtcbiAgfVxuXG4gIGZpbmRCYWNrZW5kRmFjdG9yeShiYWNrZW5kTmFtZTogc3RyaW5nKTpcbiAgICAgICgpID0+IEtlcm5lbEJhY2tlbmQgfCBQcm9taXNlPEtlcm5lbEJhY2tlbmQ+IHtcbiAgICBpZiAoIShiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5yZWdpc3RyeUZhY3RvcnlbYmFja2VuZE5hbWVdLmZhY3Rvcnk7XG4gIH1cblxuICByZWdpc3RlckJhY2tlbmQoXG4gICAgICBiYWNrZW5kTmFtZTogc3RyaW5nLFxuICAgICAgZmFjdG9yeTogKCkgPT4gS2VybmVsQmFja2VuZCB8IFByb21pc2U8S2VybmVsQmFja2VuZD4sXG4gICAgICBwcmlvcml0eSA9IDEpOiBib29sZWFuIHtcbiAgICBpZiAoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeUZhY3RvcnkpIHtcbiAgICAgIGxvZy53YXJuKFxuICAgICAgICAgIGAke2JhY2tlbmROYW1lfSBiYWNrZW5kIHdhcyBhbHJlYWR5IHJlZ2lzdGVyZWQuIGAgK1xuICAgICAgICAgIGBSZXVzaW5nIGV4aXN0aW5nIGJhY2tlbmQgZmFjdG9yeS5gKTtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgdGhpcy5yZWdpc3RyeUZhY3RvcnlbYmFja2VuZE5hbWVdID0ge2ZhY3RvcnksIHByaW9yaXR5fTtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGFzeW5jIHNldEJhY2tlbmQoYmFja2VuZE5hbWU6IHN0cmluZyk6IFByb21pc2U8Ym9vbGVhbj4ge1xuICAgIGlmICh0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiYWNrZW5kTmFtZV0gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBCYWNrZW5kIG5hbWUgJyR7YmFja2VuZE5hbWV9JyBub3QgZm91bmQgaW4gcmVnaXN0cnlgKTtcbiAgICB9XG4gICAgdGhpcy5iYWNrZW5kTmFtZSA9IGJhY2tlbmROYW1lO1xuICAgIGlmICh0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXSA9PSBudWxsKSB7XG4gICAgICB0aGlzLmJhY2tlbmRJbnN0YW5jZSA9IG51bGw7XG4gICAgICBjb25zdCB7c3VjY2VzcywgYXN5bmNJbml0fSA9IHRoaXMuaW5pdGlhbGl6ZUJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgICAgY29uc3QgcmVzdWx0ID0gYXN5bmNJbml0ID8gYXdhaXQgc3VjY2VzcyA6IHN1Y2Nlc3M7XG4gICAgICBpZiAoIXJlc3VsdCkge1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMuYmFja2VuZEluc3RhbmNlID0gdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV07XG4gICAgdGhpcy5zZXR1cFJlZ2lzdGVyZWRLZXJuZWxzKCk7XG4gICAgLy8gUmVzZXQgdGhlIHByb2ZpbGVyLlxuICAgIHRoaXMucHJvZmlsZXIgPSBuZXcgUHJvZmlsZXIodGhpcy5iYWNrZW5kSW5zdGFuY2UpO1xuXG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICBwcml2YXRlIHNldHVwUmVnaXN0ZXJlZEtlcm5lbHMoKTogdm9pZCB7XG4gICAgY29uc3Qga2VybmVscyA9IGdldEtlcm5lbHNGb3JCYWNrZW5kKHRoaXMuYmFja2VuZE5hbWUpO1xuICAgIGtlcm5lbHMuZm9yRWFjaChrZXJuZWwgPT4ge1xuICAgICAgaWYgKGtlcm5lbC5zZXR1cEZ1bmMgIT0gbnVsbCkge1xuICAgICAgICBrZXJuZWwuc2V0dXBGdW5jKHRoaXMuYmFja2VuZEluc3RhbmNlKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIHByaXZhdGUgZGlzcG9zZVJlZ2lzdGVyZWRLZXJuZWxzKGJhY2tlbmROYW1lOiBzdHJpbmcpOiB2b2lkIHtcbiAgICBjb25zdCBrZXJuZWxzID0gZ2V0S2VybmVsc0ZvckJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgIGtlcm5lbHMuZm9yRWFjaChrZXJuZWwgPT4ge1xuICAgICAgaWYgKGtlcm5lbC5kaXNwb3NlRnVuYyAhPSBudWxsKSB7XG4gICAgICAgIGtlcm5lbC5kaXNwb3NlRnVuYyh0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXSk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXMgYSBiYWNrZW5kIGJ5IGxvb2tpbmcgdXAgdGhlIGJhY2tlbmQgbmFtZSBpbiB0aGUgZmFjdG9yeVxuICAgKiByZWdpc3RyeSBhbmQgY2FsbGluZyB0aGUgZmFjdG9yeSBtZXRob2QuIFJldHVybnMgYSBib29sZWFuIHJlcHJlc2VudGluZ1xuICAgKiB3aGV0aGVyIHRoZSBpbml0aWFsaXphdGlvbiBvZiB0aGUgYmFja2VuZCBzdWNlZWRlZC4gVGhyb3dzIGFuIGVycm9yIGlmXG4gICAqIHRoZXJlIGlzIG5vIGJhY2tlbmQgaW4gdGhlIGZhY3RvcnkgcmVnaXN0cnkuXG4gICAqL1xuICBwcml2YXRlIGluaXRpYWxpemVCYWNrZW5kKGJhY2tlbmROYW1lOiBzdHJpbmcpOlxuICAgICAge3N1Y2Nlc3M6IGJvb2xlYW58UHJvbWlzZTxib29sZWFuPiwgYXN5bmNJbml0OiBib29sZWFufSB7XG4gICAgY29uc3QgcmVnaXN0cnlGYWN0b3J5RW50cnkgPSB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiYWNrZW5kTmFtZV07XG4gICAgaWYgKHJlZ2lzdHJ5RmFjdG9yeUVudHJ5ID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQ2Fubm90IGluaXRpYWxpemUgYmFja2VuZCAke2JhY2tlbmROYW1lfSwgbm8gcmVnaXN0cmF0aW9uIGZvdW5kLmApO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICBjb25zdCBiYWNrZW5kID0gcmVnaXN0cnlGYWN0b3J5RW50cnkuZmFjdG9yeSgpO1xuICAgICAgLyogVGVzdCBpZiB0aGUgZmFjdG9yeSByZXR1cm5zIGEgcHJvbWlzZS5cbiAgICAgIERvbmUgaW4gYSBtb3JlIGxpYmVyYWwgd2F5IHRoYW5cbiAgICAgIHByZXZpb3VzICdQcm9taXNlLnJlc29sdmUoYmFja2VuZCk9PT1iYWNrZW5kJ1xuICAgICAgYXMgd2UgbmVlZGVkIHRvIGFjY291bnQgZm9yIGN1c3RvbSBQcm9taXNlXG4gICAgICBpbXBsZW1lbnRhdGlvbnMgKGUuZy4gQW5ndWxhcikgKi9cbiAgICAgIGlmIChiYWNrZW5kICYmICEoYmFja2VuZCBpbnN0YW5jZW9mIEtlcm5lbEJhY2tlbmQpICYmXG4gICAgICAgICAgdHlwZW9mIGJhY2tlbmQudGhlbiA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICBjb25zdCBwcm9taXNlSWQgPSArK3RoaXMucGVuZGluZ0JhY2tlbmRJbml0SWQ7XG4gICAgICAgIGNvbnN0IHN1Y2Nlc3MgPVxuICAgICAgICAgICAgYmFja2VuZFxuICAgICAgICAgICAgICAgIC50aGVuKGJhY2tlbmRJbnN0YW5jZSA9PiB7XG4gICAgICAgICAgICAgICAgICAvLyBPdXRkYXRlZCBwcm9taXNlLiBBbm90aGVyIGJhY2tlbmQgd2FzIHNldCBpbiB0aGUgbWVhbnRpbWUuXG4gICAgICAgICAgICAgICAgICBpZiAocHJvbWlzZUlkIDwgdGhpcy5wZW5kaW5nQmFja2VuZEluaXRJZCkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXSA9IGJhY2tlbmRJbnN0YW5jZTtcbiAgICAgICAgICAgICAgICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ID0gbnVsbDtcbiAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH0pXG4gICAgICAgICAgICAgICAgLmNhdGNoKGVyciA9PiB7XG4gICAgICAgICAgICAgICAgICAvLyBPdXRkYXRlZCBwcm9taXNlLiBBbm90aGVyIGJhY2tlbmQgd2FzIHNldCBpbiB0aGUgbWVhbnRpbWUuXG4gICAgICAgICAgICAgICAgICBpZiAocHJvbWlzZUlkIDwgdGhpcy5wZW5kaW5nQmFja2VuZEluaXRJZCkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCA9IG51bGw7XG4gICAgICAgICAgICAgICAgICBsb2cud2FybihcbiAgICAgICAgICAgICAgICAgICAgICBgSW5pdGlhbGl6YXRpb24gb2YgYmFja2VuZCAke2JhY2tlbmROYW1lfSBmYWlsZWRgKTtcbiAgICAgICAgICAgICAgICAgIGxvZy53YXJuKGVyci5zdGFjayB8fCBlcnIubWVzc2FnZSk7XG4gICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ID0gc3VjY2VzcztcbiAgICAgICAgcmV0dXJuIHtzdWNjZXNzLCBhc3luY0luaXQ6IHRydWV9O1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0gPSBiYWNrZW5kIGFzIEtlcm5lbEJhY2tlbmQ7XG4gICAgICAgIHJldHVybiB7c3VjY2VzczogdHJ1ZSwgYXN5bmNJbml0OiBmYWxzZX07XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBsb2cud2FybihgSW5pdGlhbGl6YXRpb24gb2YgYmFja2VuZCAke2JhY2tlbmROYW1lfSBmYWlsZWRgKTtcbiAgICAgIGxvZy53YXJuKGVyci5zdGFjayB8fCBlcnIubWVzc2FnZSk7XG4gICAgICByZXR1cm4ge3N1Y2Nlc3M6IGZhbHNlLCBhc3luY0luaXQ6IGZhbHNlfTtcbiAgICB9XG4gIH1cblxuICByZW1vdmVCYWNrZW5kKGJhY2tlbmROYW1lOiBzdHJpbmcpOiB2b2lkIHtcbiAgICBpZiAoIShiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgJHtiYWNrZW5kTmFtZX0gYmFja2VuZCBub3QgZm91bmQgaW4gcmVnaXN0cnlgKTtcbiAgICB9XG4gICAgaWYgKHRoaXMuYmFja2VuZE5hbWUgPT09IGJhY2tlbmROYW1lICYmIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ICE9IG51bGwpIHtcbiAgICAgIC8vIFRoZXJlIGlzIGEgcGVuZGluZyBwcm9taXNlIG9mIHRoZSBiYWNrZW5kIHdlIHdhbnQgdG8gcmVtb3ZlLiBNYWtlIGl0XG4gICAgICAvLyBvYnNvbGV0ZS5cbiAgICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0SWQrKztcbiAgICB9XG5cbiAgICBpZiAoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeSkge1xuICAgICAgdGhpcy5kaXNwb3NlUmVnaXN0ZXJlZEtlcm5lbHMoYmFja2VuZE5hbWUpO1xuICAgICAgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0uZGlzcG9zZSgpO1xuICAgICAgZGVsZXRlIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdO1xuICAgIH1cblxuICAgIGRlbGV0ZSB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiYWNrZW5kTmFtZV07XG5cbiAgICAvLyBVbnNldCB0aGUgYmFja2VuZCBpZiBpdCBpcyBhY3RpdmUuXG4gICAgaWYgKHRoaXMuYmFja2VuZE5hbWUgPT09IGJhY2tlbmROYW1lKSB7XG4gICAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCA9IG51bGw7XG4gICAgICB0aGlzLmJhY2tlbmROYW1lID0gbnVsbDtcbiAgICAgIHRoaXMuYmFja2VuZEluc3RhbmNlID0gbnVsbDtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGdldFNvcnRlZEJhY2tlbmRzKCk6IHN0cmluZ1tdIHtcbiAgICBpZiAoT2JqZWN0LmtleXModGhpcy5yZWdpc3RyeUZhY3RvcnkpLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdObyBiYWNrZW5kIGZvdW5kIGluIHJlZ2lzdHJ5LicpO1xuICAgIH1cbiAgICByZXR1cm4gT2JqZWN0LmtleXModGhpcy5yZWdpc3RyeUZhY3RvcnkpLnNvcnQoKGE6IHN0cmluZywgYjogc3RyaW5nKSA9PiB7XG4gICAgICAvLyBIaWdoZXN0IHByaW9yaXR5IGNvbWVzIGZpcnN0LlxuICAgICAgcmV0dXJuIHRoaXMucmVnaXN0cnlGYWN0b3J5W2JdLnByaW9yaXR5IC1cbiAgICAgICAgICB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVthXS5wcmlvcml0eTtcbiAgICB9KTtcbiAgfVxuXG4gIHByaXZhdGUgaW5pdGlhbGl6ZUJhY2tlbmRzQW5kUmV0dXJuQmVzdCgpOlxuICAgICAge25hbWU6IHN0cmluZywgYXN5bmNJbml0OiBib29sZWFufSB7XG4gICAgY29uc3Qgc29ydGVkQmFja2VuZHMgPSB0aGlzLmdldFNvcnRlZEJhY2tlbmRzKCk7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNvcnRlZEJhY2tlbmRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBiYWNrZW5kTmFtZSA9IHNvcnRlZEJhY2tlbmRzW2ldO1xuICAgICAgY29uc3Qge3N1Y2Nlc3MsIGFzeW5jSW5pdH0gPSB0aGlzLmluaXRpYWxpemVCYWNrZW5kKGJhY2tlbmROYW1lKTtcbiAgICAgIGlmIChhc3luY0luaXQgfHwgc3VjY2Vzcykge1xuICAgICAgICByZXR1cm4ge25hbWU6IGJhY2tlbmROYW1lLCBhc3luY0luaXR9O1xuICAgICAgfVxuICAgIH1cbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBDb3VsZCBub3QgaW5pdGlhbGl6ZSBhbnkgYmFja2VuZHMsIGFsbCBiYWNrZW5kIGluaXRpYWxpemF0aW9ucyBgICtcbiAgICAgICAgYGZhaWxlZC5gKTtcbiAgfVxuXG4gIG1vdmVEYXRhKGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQsIGRhdGFJZDogRGF0YUlkKSB7XG4gICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCBzcmNCYWNrZW5kID0gaW5mby5iYWNrZW5kO1xuICAgIGNvbnN0IHZhbHVlcyA9IHRoaXMucmVhZFN5bmMoZGF0YUlkKTtcbiAgICBjb25zdCByZWZDb3VudCA9IHNyY0JhY2tlbmQucmVmQ291bnQoZGF0YUlkKTtcbiAgICAvLyBEZWxldGUgdGhlIHRlbnNvciBmcm9tIHRoZSBvbGQgYmFja2VuZCBhbmQgbW92ZSBpdCB0byB0aGUgbmV3XG4gICAgLy8gYmFja2VuZC5cbiAgICBzcmNCYWNrZW5kLmRpc3Bvc2VEYXRhKGRhdGFJZCwgdHJ1ZSk7XG4gICAgaW5mby5iYWNrZW5kID0gYmFja2VuZDtcbiAgICBiYWNrZW5kLm1vdmUoZGF0YUlkLCB2YWx1ZXMsIGluZm8uc2hhcGUsIGluZm8uZHR5cGUsIHJlZkNvdW50KTtcbiAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgIC8vIFRyYWNrIHRoZSBudW1iZXIgb2YgbW92ZXMgZHVyaW5nIGEga2VybmVsIGV4ZWN1dGlvbiB0byBjb3JyZWN0bHlcbiAgICAgIC8vIGRldGVjdCBtZW1vcnkgbGVha3MuXG4gICAgICB0aGlzLnN0YXRlLm51bURhdGFNb3Zlc1N0YWNrW3RoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2subGVuZ3RoIC0gMV0rKztcbiAgICB9XG4gIH1cblxuICB0aWR5PFQgZXh0ZW5kcyBUZW5zb3JDb250YWluZXI+KG5hbWVPckZuOiBzdHJpbmd8U2NvcGVGbjxUPiwgZm4/OiBTY29wZUZuPFQ+KTpcbiAgICAgIFQge1xuICAgIGxldCBuYW1lOiBzdHJpbmcgPSBudWxsO1xuICAgIGlmIChmbiA9PSBudWxsKSB7XG4gICAgICAvLyBDYWxsZWQgd2l0aCBvbmx5IDEgYXJndW1lbnQuXG4gICAgICBpZiAodHlwZW9mIG5hbWVPckZuICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignUGxlYXNlIHByb3ZpZGUgYSBmdW5jdGlvbiB0byB0aWR5KCknKTtcbiAgICAgIH1cbiAgICAgIGZuID0gbmFtZU9yRm47XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIENhbGxlZCB3aXRoIDIgYXJndW1lbnRzLlxuICAgICAgaWYgKHR5cGVvZiBuYW1lT3JGbiAhPT0gJ3N0cmluZycgJiYgIShuYW1lT3JGbiBpbnN0YW5jZW9mIFN0cmluZykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ1doZW4gY2FsbGluZyB3aXRoIHR3byBhcmd1bWVudHMsIHRoZSBmaXJzdCBhcmd1bWVudCAnICtcbiAgICAgICAgICAgICd0byB0aWR5KCkgbXVzdCBiZSBhIHN0cmluZycpO1xuICAgICAgfVxuICAgICAgaWYgKHR5cGVvZiBmbiAhPT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnV2hlbiBjYWxsaW5nIHdpdGggdHdvIGFyZ3VtZW50cywgdGhlIDJuZCBhcmd1bWVudCAnICtcbiAgICAgICAgICAgICd0byB0aWR5KCkgbXVzdCBiZSBhIGZ1bmN0aW9uJyk7XG4gICAgICB9XG4gICAgICBuYW1lID0gbmFtZU9yRm4gYXMgc3RyaW5nO1xuICAgICAgLy8gVE9ETyhuc3Rob3JhdCxzbWlsa292KTogRG8gb3BlcmF0aW9uIGxvZ2dpbmcgYW5kIHBlcmZvcm1hbmNlXG4gICAgICAvLyBwcm9maWxpbmcuXG4gICAgfVxuICAgIGxldCByZXN1bHQ6IFQ7XG4gICAgcmV0dXJuIHRoaXMuc2NvcGVkUnVuKFxuICAgICAgICAoKSA9PiB0aGlzLnN0YXJ0U2NvcGUobmFtZSksICgpID0+IHRoaXMuZW5kU2NvcGUocmVzdWx0KSwgKCkgPT4ge1xuICAgICAgICAgIHJlc3VsdCA9IGZuKCk7XG4gICAgICAgICAgaWYgKHJlc3VsdCBpbnN0YW5jZW9mIFByb21pc2UpIHtcbiAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Nhbm5vdCByZXR1cm4gYSBQcm9taXNlIGluc2lkZSBvZiB0aWR5LicpO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm4gcmVzdWx0O1xuICAgICAgICB9KTtcbiAgfVxuXG4gIHByaXZhdGUgc2NvcGVkUnVuPFQ+KHN0YXJ0OiAoKSA9PiB2b2lkLCBlbmQ6ICgpID0+IHZvaWQsIGY6ICgpID0+IFQpOiBUIHtcbiAgICBzdGFydCgpO1xuICAgIHRyeSB7XG4gICAgICBjb25zdCByZXMgPSBmKCk7XG4gICAgICBlbmQoKTtcbiAgICAgIHJldHVybiByZXM7XG4gICAgfSBjYXRjaCAoZXgpIHtcbiAgICAgIGVuZCgpO1xuICAgICAgdGhyb3cgZXg7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBzdGF0aWMgbmV4dFRlbnNvcklkID0gMDtcbiAgcHJpdmF0ZSBuZXh0VGVuc29ySWQoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gRW5naW5lLm5leHRUZW5zb3JJZCsrO1xuICB9XG5cbiAgcHJpdmF0ZSBzdGF0aWMgbmV4dFZhcmlhYmxlSWQgPSAwO1xuICBwcml2YXRlIG5leHRWYXJpYWJsZUlkKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIEVuZ2luZS5uZXh0VmFyaWFibGVJZCsrO1xuICB9XG5cbiAgLyoqXG4gICAqIFRoaXMgbWV0aG9kIGlzIGNhbGxlZCBpbnN0ZWFkIG9mIHRoZSBwdWJsaWMtZmFjaW5nIHRlbnNvci5jbG9uZSgpIHdoZW5cbiAgICogc2F2aW5nIGEgdGVuc29yIGZvciBiYWNrd2FyZHMgcGFzcy4gSXQgbWFrZXMgc3VyZSB0byBhZGQgdGhlIGNsb25lXG4gICAqIG9wZXJhdGlvbiB0byB0aGUgdGFwZSByZWdhcmRsZXNzIG9mIGJlaW5nIGNhbGxlZCBpbnNpZGUgYSBrZXJuZWxcbiAgICogZXhlY3V0aW9uLlxuICAgKi9cbiAgcHJpdmF0ZSBjbG9uZSh4OiBUZW5zb3IpOiBUZW5zb3Ige1xuICAgIGNvbnN0IHk6IFRlbnNvciA9IEVOR0lORS5ydW5LZXJuZWwoSWRlbnRpdHksIHt4fSBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCk7XG4gICAgY29uc3QgaW5wdXRzID0ge3h9O1xuICAgIGNvbnN0IGdyYWQgPSAoZHk6IFRlbnNvcikgPT4gKHtcbiAgICAgIHg6ICgpID0+IHtcbiAgICAgICAgY29uc3QgZHR5cGUgPSAnZmxvYXQzMic7XG4gICAgICAgIGNvbnN0IGdyYWRJbnB1dHMgPSB7eDogZHl9O1xuICAgICAgICBjb25zdCBhdHRycyA9IHtkdHlwZX07XG5cbiAgICAgICAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICAgQ2FzdCwgZ3JhZElucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICAgICAgICAgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKSBhcyBUZW5zb3I7XG4gICAgICB9XG4gICAgfSk7XG4gICAgY29uc3Qgc2F2ZWQ6IFRlbnNvcltdID0gW107XG4gICAgdGhpcy5hZGRUYXBlTm9kZSh0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLm5hbWUsIGlucHV0cywgW3ldLCBncmFkLCBzYXZlZCwge30pO1xuICAgIHJldHVybiB5O1xuICB9XG5cbiAgLyoqXG4gICAqIEV4ZWN1dGUgYSBrZXJuZWwgd2l0aCB0aGUgZ2l2ZW4gbmFtZSBhbmQgcmV0dXJuIHRoZSBvdXRwdXQgdGVuc29yLlxuICAgKlxuICAgKiBAcGFyYW0ga2VybmVsTmFtZSBUaGUgbmFtZSBvZiB0aGUga2VybmVsIHRvIGV4ZWN1dGUuXG4gICAqIEBwYXJhbSBpbnB1dHMgQSBtYXAgb2YgaW5wdXQgbmFtZXMgdG8gdGVuc29ycy5cbiAgICogQHBhcmFtIGF0dHJzIEEgbWFwIG9mIGF0dHJpYnV0ZSBuYW1lcyB0byB0aGVpciB2YWx1ZXMuIEFuIGF0dHJpYnV0ZSBpcyBhXG4gICAqICAgICBwcmltaXRpdmUgKG5vbi10ZW5zb3IpIGlucHV0IHRvIHRoZSBrZXJuZWwuXG4gICAqIEBwYXJhbSBpbnB1dHNUb1NhdmUgQSBsaXN0IG9mIHRlbnNvcnMsIGlucHV0cyB0byBzYXZlIGZvciB0aGUgYmFja3Byb3BcbiAgICogICAgIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gb3V0cHV0c1RvU2F2ZSBBIGxpc3Qgb2YgYm9vbGVhbnMsIHNwZWNpZnlpbmcgd2hpY2ggb3V0cHV0IHRvIHNhdmVcbiAgICogICAgIGZvciB0aGUgYmFja3Byb3AgY29tcHV0YXRpb24uIFRoZXNlIGFyZSBib29sZWFucyBzaW5jZSB0aGUgb3V0cHV0XG4gICAqIHRlbnNvcnMgYXJlIG5vdCB2aXNpYmxlIHRvIHRoZSB1c2VyLlxuICAgKi9cbiAgcnVuS2VybmVsPFQgZXh0ZW5kcyBUZW5zb3J8VGVuc29yW10+KFxuICAgICAga2VybmVsTmFtZTogc3RyaW5nLCBpbnB1dHM6IE5hbWVkVGVuc29yTWFwLCBhdHRycz86IE5hbWVkQXR0ck1hcCk6IFQge1xuICAgIGlmICh0aGlzLmJhY2tlbmROYW1lID09IG51bGwpIHtcbiAgICAgIC8vIGJhY2tlbmQgaGFzIG5vdCBiZWVuIGluaXRpYWxpemVkIHlldCAoYmFja2VuZCBpbml0aWFsaXphdGlvbiBpcyBsYXp5XG4gICAgICAvLyBjYW4gYmUgZGVmZXJyZWQgdW50aWwgYW4gb3AvIGtlcm5lbCBpcyBydW4pLlxuICAgICAgLy8gVGhlIGJlbG93IGdldHRlciBoYXMgc2lkZSBlZmZlY3RzIHRoYXQgd2lsbCB0cnkgdG8gaW5pdGlhbGl6ZSB0aGVcbiAgICAgIC8vIGJhY2tlbmQgYW5kIHNldCBwcm9wZXJ0aWVzIGxpa2UgdGhpcy5iYWNrZW5kTmFtZVxuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bnVzZWQtZXhwcmVzc2lvblxuICAgICAgdGhpcy5iYWNrZW5kO1xuICAgIH1cbiAgICBjb25zdCBoYXNLZXJuZWwgPSBnZXRLZXJuZWwoa2VybmVsTmFtZSwgdGhpcy5iYWNrZW5kTmFtZSkgIT0gbnVsbDtcbiAgICBpZiAoIWhhc0tlcm5lbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBLZXJuZWwgJyR7a2VybmVsTmFtZX0nIG5vdCByZWdpc3RlcmVkIGZvciBiYWNrZW5kICcke1xuICAgICAgICAgIHRoaXMuYmFja2VuZE5hbWV9J2ApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5ydW5LZXJuZWxGdW5jKHtrZXJuZWxOYW1lLCBpbnB1dHMsIGF0dHJzfSk7XG4gIH1cblxuICBwcml2YXRlIHNob3VsZENoZWNrRm9yTWVtTGVha3MoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuRU5WLmdldEJvb2woJ0lTX1RFU1QnKTtcbiAgfVxuXG4gIHByaXZhdGUgY2hlY2tLZXJuZWxGb3JNZW1MZWFrKFxuICAgICAga2VybmVsTmFtZTogc3RyaW5nLCBudW1EYXRhSWRzQmVmb3JlOiBudW1iZXIsXG4gICAgICBvdXRJbmZvczogVGVuc29ySW5mb1tdKTogdm9pZCB7XG4gICAgY29uc3QgbnVtRGF0YUlkc0FmdGVyID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcblxuICAgIC8vIENvdW50IHRoZSBudW1iZXIgb2YgZGF0YSBpZHMgYXNzb2NpYXRlZCB3aXRoIHRoZSByZXN1bHQgb2YgdGhlIGtlcm5lbC5cbiAgICBsZXQgbnVtT3V0cHV0RGF0YUlkcyA9IDA7XG4gICAgb3V0SW5mb3MuZm9yRWFjaChpbmZvID0+IHtcbiAgICAgIC8vIENvbXBsZXggbnVtYmVycyBhbGxvY2F0ZSAzIGRhdGEgaWRzLCBvbmUgZm9yICdyZWFsJywgb25lIGZvclxuICAgICAgLy8gJ2ltYWdpbmFyeScsIGFuZCBvbmUgZm9yIHRoZSBjb250YWluZXIgdGhhdCBob2xkcyB0aGUgZm9ybWVyIHR3by5cbiAgICAgIG51bU91dHB1dERhdGFJZHMgKz0gKGluZm8uZHR5cGUgPT09ICdjb21wbGV4NjQnID8gMyA6IDEpO1xuICAgIH0pO1xuXG4gICAgLy8gQWNjb3VudCBmb3IgdGhlIG51bWJlciBvZiBtb3ZlcyBkdXJpbmcga2VybmVsIGV4ZWN1dGlvbi4gQSBcImRhdGEgbW92ZVwiXG4gICAgLy8gY2FuIGhhcHBlbiBpbiB0aGUgbWlkZGxlIG9mIGEga2VybmVsIGV4ZWN1dGlvbiwgcGxhY2luZyBhIG5ldyAoa2V5LHZhbHVlKVxuICAgIC8vIHBhaXIgaW4gdGhlIGRhdGEgc3RvcmFnZS4gU2luY2UgZGF0YSBtb3ZlcyBoYXZlIG5ldCB6ZXJvIGVmZmVjdCAod2VcbiAgICAvLyBhbHdheXMgcmVtb3ZlIHRoZSBkYXRhIGZyb20gdGhlIG9sZCBiYWNrZW5kKSwgd2UgaGF2ZSB0byBjYW5jZWwgdGhlbSBvdXRcbiAgICAvLyB3aGVuIGRldGVjdGluZyBtZW1vcnkgbGVha3MuXG4gICAgY29uc3QgbnVtTW92ZXMgPVxuICAgICAgICB0aGlzLnN0YXRlLm51bURhdGFNb3Zlc1N0YWNrW3RoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2subGVuZ3RoIC0gMV07XG4gICAgY29uc3QgZGF0YUlkc0xlYWtlZCA9XG4gICAgICAgIG51bURhdGFJZHNBZnRlciAtIG51bURhdGFJZHNCZWZvcmUgLSBudW1PdXRwdXREYXRhSWRzIC0gbnVtTW92ZXM7XG4gICAgaWYgKGRhdGFJZHNMZWFrZWQgPiAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEJhY2tlbmQgJyR7dGhpcy5iYWNrZW5kTmFtZX0nIGhhcyBhbiBpbnRlcm5hbCBtZW1vcnkgbGVhayBgICtcbiAgICAgICAgICBgKCR7ZGF0YUlkc0xlYWtlZH0gZGF0YSBpZHMpIGFmdGVyIHJ1bm5pbmcgJyR7a2VybmVsTmFtZX0nYCk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIEludGVybmFsIGhlbHBlciBtZXRob2QgdG8gZXhlY3V0ZSBhIGtlcm5lbCBGdW5jXG4gICAqXG4gICAqIFVzZSBgcnVuS2VybmVsYCB0byBleGVjdXRlIGtlcm5lbHMgZnJvbSBvdXRzaWRlIG9mIGVuZ2luZS5cbiAgICovXG4gIHByaXZhdGUgcnVuS2VybmVsRnVuYzxUIGV4dGVuZHMgVGVuc29yfFRlbnNvcltdLCBJIGV4dGVuZHMgTmFtZWRUZW5zb3JNYXA+KFxuICAgICAga2VybmVsUGFyYW1zOiBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJPnxcbiAgICAgIEN1c3RvbUdyYWRLZXJuZWxJbnZvY2F0aW9uPFQsIEk+KTogVCB7XG4gICAgbGV0IG91dHB1dHM6IFRlbnNvcltdO1xuICAgIGxldCBzYXZlZDogVGVuc29yW10gPSBbXTtcbiAgICBjb25zdCBpc1RhcGVPbiA9IHRoaXMuaXNUYXBlT24oKTtcblxuICAgIGNvbnN0IHN0YXJ0aW5nQnl0ZWNvdW50ID0gdGhpcy5zdGF0ZS5udW1CeXRlcztcbiAgICBjb25zdCBzdGFydGluZ051bVRlbnNvcnMgPSB0aGlzLnN0YXRlLm51bVRlbnNvcnM7XG5cbiAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgIHRoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2sucHVzaCgwKTtcbiAgICB9XG5cbiAgICBsZXQga2VybmVsRnVuYzogKCkgPT4gVGVuc29yW107XG4gICAgaWYgKHRoaXMuYmFja2VuZE5hbWUgPT0gbnVsbCkge1xuICAgICAgLy8gYmFja2VuZCBoYXMgbm90IGJlZW4gaW5pdGlhbGl6ZWQgeWV0IChiYWNrZW5kIGluaXRpYWxpemF0aW9uIGlzIGxhenlcbiAgICAgIC8vIGNhbiBiZSBkZWZlcnJlZCB1bnRpbCBhbiBvcC8ga2VybmVsIGlzIHJ1bikuXG4gICAgICAvLyBUaGUgYmVsb3cgZ2V0dGVyIGhhcyBzaWRlIGVmZmVjdHMgdGhhdCB3aWxsIHRyeSB0byBpbml0aWFsaXplIHRoZVxuICAgICAgLy8gYmFja2VuZCBhbmQgc2V0IHByb3BlcnRpZXMgbGlrZSB0aGlzLmJhY2tlbmROYW1lXG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgICB0aGlzLmJhY2tlbmQ7XG4gICAgfVxuXG4gICAgbGV0IG91dDogVGVuc29ySW5mb3xUZW5zb3JJbmZvW107XG5cbiAgICBjb25zdCBrZXJuZWxPclNjb3BlTmFtZSA9IGlzUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb24oa2VybmVsUGFyYW1zKSA/XG4gICAgICAgIGtlcm5lbFBhcmFtcy5rZXJuZWxOYW1lIDpcbiAgICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZSAhPSBudWxsID8gdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS5uYW1lIDogJyc7XG5cbiAgICAvLyBDcmVhdGUgdGhlIGtlcm5lbEZ1bmMgZnJvbSBlaXRoZXIgYSByZWdpc3RlcmVkIGtlcm5lbCBPUiBwYXNzZWQgaW5cbiAgICAvLyBmb3J3YXJkL2JhY2t3YXJkIGZ1bmN0aW9ucyAodXNlZCBieSBjdXN0b20gZ3JhZCkuIEluIHRoaXMgY29udGV4dCBhXG4gICAgLy8ga2VybmVsRnVuYyB3cmFwcyBhIGtlcm5lbCBpbXBsZW1lbnRhdGlvbiB3aXRoIHNvbWUgYm9va2tlZXBpbmcuXG5cbiAgICBpZiAoaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbihrZXJuZWxQYXJhbXMpKSB7XG4gICAgICBjb25zdCB7a2VybmVsTmFtZSwgaW5wdXRzLCBhdHRyc30gPSBrZXJuZWxQYXJhbXM7XG4gICAgICBpZiAodGhpcy5iYWNrZW5kTmFtZSA9PSBudWxsKSB7XG4gICAgICAgIC8vIGJhY2tlbmQgaGFzIG5vdCBiZWVuIGluaXRpYWxpemVkIHlldCAoYmFja2VuZCBpbml0aWFsaXphdGlvbiBpcyBsYXp5XG4gICAgICAgIC8vIGNhbiBiZSBkZWZlcnJlZCB1bnRpbCBhbiBvcC8ga2VybmVsIGlzIHJ1bikuXG4gICAgICAgIC8vIFRoZSBiZWxvdyBnZXR0ZXIgaGFzIHNpZGUgZWZmZWN0cyB0aGF0IHdpbGwgdHJ5IHRvIGluaXRpYWxpemUgdGhlXG4gICAgICAgIC8vIGJhY2tlbmQgYW5kIHNldCBwcm9wZXJ0aWVzIGxpa2UgdGhpcy5iYWNrZW5kTmFtZVxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgICAgIHRoaXMuYmFja2VuZDtcbiAgICAgIH1cbiAgICAgIGNvbnN0IGtlcm5lbCA9IGdldEtlcm5lbChrZXJuZWxOYW1lLCB0aGlzLmJhY2tlbmROYW1lKTtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGtlcm5lbCAhPSBudWxsLFxuICAgICAgICAgICgpID0+IGBDYW5ub3QgZmluZCByZWdpc3RlcmVkIGtlcm5lbCAnJHtrZXJuZWxOYW1lfScgZm9yIGJhY2tlbmQgJyR7XG4gICAgICAgICAgICAgIHRoaXMuYmFja2VuZE5hbWV9J2ApO1xuXG4gICAgICBrZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBudW1EYXRhSWRzQmVmb3JlID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcbiAgICAgICAgb3V0ID0ga2VybmVsLmtlcm5lbEZ1bmMoe2lucHV0cywgYXR0cnMsIGJhY2tlbmQ6IHRoaXMuYmFja2VuZH0pO1xuICAgICAgICBjb25zdCBvdXRJbmZvcyA9IEFycmF5LmlzQXJyYXkob3V0KSA/IG91dCA6IFtvdXRdO1xuICAgICAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgICAgICB0aGlzLmNoZWNrS2VybmVsRm9yTWVtTGVhayhrZXJuZWxOYW1lLCBudW1EYXRhSWRzQmVmb3JlLCBvdXRJbmZvcyk7XG4gICAgICAgIH1cblxuICAgICAgICBjb25zdCBvdXRUZW5zb3JzID0gb3V0SW5mb3MubWFwKChvdXRJbmZvOiBUZW5zb3JJbmZvfFRlbnNvcikgPT4ge1xuICAgICAgICAgIC8vIHRvZG8gKHlhc3NvZ2JhKSByZW1vdmUgdGhpcyBvcHRpb24gKFRlbnNvcikgd2hlbiBub2RlIGJhY2tlbmRcbiAgICAgICAgICAvLyBtZXRob2RzIGhhdmUgYmVlbiBtb2R1bGFyaXplZCBhbmQgdGhleSBhbGwgcmV0dXJuIHRlbnNvckluZm8uXG4gICAgICAgICAgLy8gVGVuc29ySW5mb3MgZG8gbm90IGhhdmUgYSByYW5rIGF0dHJpYnV0ZS5cbiAgICAgICAgICBpZiAoKG91dEluZm8gYXMgVGVuc29yKS5yYW5rICE9IG51bGwpIHtcbiAgICAgICAgICAgIHJldHVybiBvdXRJbmZvIGFzIFRlbnNvcjtcbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3Qge2RhdGFJZCwgc2hhcGUsIGR0eXBlfSA9IG91dEluZm8gYXMgVGVuc29ySW5mbztcbiAgICAgICAgICByZXR1cm4gdGhpcy5tYWtlVGVuc29yRnJvbURhdGFJZChkYXRhSWQsIHNoYXBlLCBkdHlwZSk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIFNhdmUgYW55IHJlcXVpcmVkIGlucHV0cyBhbmQgb3V0cHV0cy5cblxuICAgICAgICAvLyBEbyBub3Qgc2F2ZSB1bmxlc3Mgd2UgYXJlIHJlY29yZGluZyB0byB0aGUgdGFwZS4gT3RoZXJ3aXNlIGl0IHdvdWxkXG4gICAgICAgIC8vIGNhdXNlIGEgbWVtIGxlYWsgc2luY2UgdGhlcmUgd291bGQgYmUgbm8gYmFja3Byb3AgZm9yIHRoZXNlIHRlbnNvcnNcbiAgICAgICAgLy8gKHdoaWNoIHdvdWxkIG90aGVyd2lzZSBkaXNwb3NlIHRoZW0pLlxuICAgICAgICBpZiAoaXNUYXBlT24pIHtcbiAgICAgICAgICBjb25zdCB0ZW5zb3JzVG9TYXZlID1cbiAgICAgICAgICAgICAgdGhpcy5nZXRUZW5zb3JzRm9yR3JhZGllbnQoa2VybmVsTmFtZSwgaW5wdXRzLCBvdXRUZW5zb3JzKTtcbiAgICAgICAgICBzYXZlZCA9IHRoaXMuc2F2ZVRlbnNvcnNGb3JCYWNrd2FyZE1vZGUodGVuc29yc1RvU2F2ZSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG91dFRlbnNvcnM7XG4gICAgICB9O1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB7Zm9yd2FyZEZ1bmN9ID0ga2VybmVsUGFyYW1zO1xuICAgICAgLy8gUnVubmluZyBhIGN1c3RvbUdyYWQgb3AuXG4gICAgICBjb25zdCBzYXZlRnVuYzogR3JhZFNhdmVGdW5jID0gKHRlbnNvcnMpID0+IHtcbiAgICAgICAgLy8gRG8gbm90IHNhdmUgdW5sZXNzIHdlIGFyZSByZWNvcmRpbmcgdG8gdGhlIHRhcGUuIE90aGVyd2lzZSBpdCB3b3VsZFxuICAgICAgICAvLyBjYXVzZSBhIG1lbSBsZWFrIHNpbmNlIHdlIHdvdWxkIG5ldmVyIHJ1biBiYWNrcHJvcCwgd2hpY2ggZGlzcG9zZXNcbiAgICAgICAgLy8gdGhlIGtlcHQgdGVuc29ycy5cbiAgICAgICAgaWYgKCFpc1RhcGVPbikge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBzYXZlZCA9IHRlbnNvcnMubWFwKHRlbnNvciA9PiB0aGlzLmtlZXAodGhpcy5jbG9uZSh0ZW5zb3IpKSk7XG4gICAgICB9O1xuXG4gICAgICBrZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBudW1EYXRhSWRzQmVmb3JlID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcbiAgICAgICAgb3V0ID0gdGhpcy50aWR5KCgpID0+IGZvcndhcmRGdW5jKHRoaXMuYmFja2VuZCwgc2F2ZUZ1bmMpKTtcbiAgICAgICAgY29uc3Qgb3V0cyA9IChBcnJheS5pc0FycmF5KG91dCkgPyBvdXQgOiBbb3V0XSkgYXMgVGVuc29yW107XG4gICAgICAgIGlmICh0aGlzLnNob3VsZENoZWNrRm9yTWVtTGVha3MoKSkge1xuICAgICAgICAgIC8vIFNjb3BlIG5hbWUgaXMgdXNlZCB0byBwcmludCBhIG1vcmUgaGVscGZ1bCBlcnJvciBtZXNzYWdlIGlmIG5lZWRlZC5cbiAgICAgICAgICB0aGlzLmNoZWNrS2VybmVsRm9yTWVtTGVhayhrZXJuZWxPclNjb3BlTmFtZSwgbnVtRGF0YUlkc0JlZm9yZSwgb3V0cyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG91dHM7XG4gICAgICB9O1xuICAgIH1cblxuICAgIC8vXG4gICAgLy8gUnVuIHRoZSBrZXJuZWxGdW5jLiBPcHRpb25hbGx5IHByb2ZpbGluZyBpdC5cbiAgICAvL1xuICAgIGNvbnN0IHtpbnB1dHMsIGF0dHJzfSA9IGtlcm5lbFBhcmFtcztcbiAgICBjb25zdCBiYWNrd2FyZHNGdW5jID0gaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbihrZXJuZWxQYXJhbXMpID9cbiAgICAgICAgbnVsbCA6XG4gICAgICAgIGtlcm5lbFBhcmFtcy5iYWNrd2FyZHNGdW5jO1xuXG4gICAgbGV0IGtlcm5lbFByb2ZpbGU6IEtlcm5lbFByb2ZpbGU7XG4gICAgdGhpcy5zY29wZWRSdW4oXG4gICAgICAgIC8vIFN0b3AgcmVjb3JkaW5nIHRvIGEgdGFwZSB3aGVuIHJ1bm5pbmcgYSBrZXJuZWwuXG4gICAgICAgICgpID0+IHRoaXMuc3RhdGUua2VybmVsRGVwdGgrKywgKCkgPT4gdGhpcy5zdGF0ZS5rZXJuZWxEZXB0aC0tLCAoKSA9PiB7XG4gICAgICAgICAgaWYgKCF0aGlzLkVOVi5nZXRCb29sKCdERUJVRycpICYmICF0aGlzLnN0YXRlLnByb2ZpbGluZykge1xuICAgICAgICAgICAgb3V0cHV0cyA9IGtlcm5lbEZ1bmMoKTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAga2VybmVsUHJvZmlsZSA9IHRoaXMucHJvZmlsZXIucHJvZmlsZUtlcm5lbChcbiAgICAgICAgICAgICAgICBrZXJuZWxPclNjb3BlTmFtZSwgaW5wdXRzLCAoKSA9PiBrZXJuZWxGdW5jKCkpO1xuICAgICAgICAgICAgaWYgKHRoaXMuRU5WLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgICAgICAgICAgdGhpcy5wcm9maWxlci5sb2dLZXJuZWxQcm9maWxlKGtlcm5lbFByb2ZpbGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgb3V0cHV0cyA9IGtlcm5lbFByb2ZpbGUub3V0cHV0cztcbiAgICAgICAgICB9XG4gICAgICAgIH0pO1xuXG4gICAgaWYgKGlzVGFwZU9uKSB7XG4gICAgICB0aGlzLmFkZFRhcGVOb2RlKFxuICAgICAgICAgIGtlcm5lbE9yU2NvcGVOYW1lLCBpbnB1dHMsIG91dHB1dHMsIGJhY2t3YXJkc0Z1bmMsIHNhdmVkLCBhdHRycyk7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMuc3RhdGUucHJvZmlsaW5nKSB7XG4gICAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUua2VybmVscy5wdXNoKHtcbiAgICAgICAgbmFtZToga2VybmVsT3JTY29wZU5hbWUsXG4gICAgICAgIGJ5dGVzQWRkZWQ6IHRoaXMuc3RhdGUubnVtQnl0ZXMgLSBzdGFydGluZ0J5dGVjb3VudCxcbiAgICAgICAgdG90YWxCeXRlc1NuYXBzaG90OiB0aGlzLnN0YXRlLm51bUJ5dGVzLFxuICAgICAgICB0ZW5zb3JzQWRkZWQ6IHRoaXMuc3RhdGUubnVtVGVuc29ycyAtIHN0YXJ0aW5nTnVtVGVuc29ycyxcbiAgICAgICAgdG90YWxUZW5zb3JzU25hcHNob3Q6IHRoaXMuc3RhdGUubnVtVGVuc29ycyxcbiAgICAgICAgaW5wdXRTaGFwZXM6IE9iamVjdC5rZXlzKGlucHV0cykubWFwKFxuICAgICAgICAgICAga2V5ID0+IGlucHV0c1trZXldICE9IG51bGwgPyBpbnB1dHNba2V5XS5zaGFwZSA6IG51bGwpLFxuICAgICAgICBvdXRwdXRTaGFwZXM6IG91dHB1dHMubWFwKGl0ZW0gPT4gaXRlbS5zaGFwZSksXG4gICAgICAgIGtlcm5lbFRpbWVNczoga2VybmVsUHJvZmlsZS50aW1lTXMsXG4gICAgICAgIGV4dHJhSW5mbzoga2VybmVsUHJvZmlsZS5leHRyYUluZm9cbiAgICAgIH0pO1xuICAgIH1cbiAgICByZXR1cm4gKEFycmF5LmlzQXJyYXkob3V0KSA/IG91dHB1dHMgOiBvdXRwdXRzWzBdKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIFNhdmVzIHRlbnNvcnMgdXNlZCBpbiBmb3J3YXJkIG1vZGUgZm9yIHVzZSBpbiBiYWNrd2FyZCBtb2RlLlxuICAgKlxuICAgKiBAcGFyYW0gdGVuc29ycyB0aGUgbGlzdCBvZiB0ZW5zb3JzIHRvIHNhdmUuXG4gICAqL1xuICBwcml2YXRlIHNhdmVUZW5zb3JzRm9yQmFja3dhcmRNb2RlKHRlbnNvcnM6IFRlbnNvcltdKTogVGVuc29yW10ge1xuICAgIGNvbnN0IHNhdmVkID0gdGVuc29ycy5tYXAodGVuc29yID0+IHRoaXMua2VlcCh0aGlzLmNsb25lKHRlbnNvcikpKTtcbiAgICByZXR1cm4gc2F2ZWQ7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBhIGxpc3Qgb2YgdGVuc29ycyB0byBzYXZlIGZvciBhIGdpdmVuIGdyYWRpZW50IGNhbGN1bGF0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0ga2VybmVsTmFtZSBuYW1lIG9mIGtlcm5lbCB0byBsb29rIHVwIGdyYWRpZW50IGZvci5cbiAgICogQHBhcmFtIGlucHV0cyBhIG1hcCBvZiBpbnB1dCB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gb3V0cHV0cyBhbiBhcnJheSBvZiBvdXRwdXQgdGVuc29ycyBmcm9tIGZvcndhcmQgbW9kZSBvZiBrZXJuZWwuXG4gICAqL1xuICBwcml2YXRlIGdldFRlbnNvcnNGb3JHcmFkaWVudChcbiAgICAgIGtlcm5lbE5hbWU6IHN0cmluZywgaW5wdXRzOiBOYW1lZFRlbnNvck1hcCxcbiAgICAgIG91dHB1dHM6IFRlbnNvcltdKTogVGVuc29yW118bnVsbCB7XG4gICAgY29uc3QgZ3JhZENvbmZpZyA9IGdldEdyYWRpZW50KGtlcm5lbE5hbWUpO1xuICAgIGlmIChncmFkQ29uZmlnICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IGlucHV0c1RvU2F2ZTogc3RyaW5nW10gPSBncmFkQ29uZmlnLmlucHV0c1RvU2F2ZSB8fCBbXTtcbiAgICAgIGNvbnN0IG91dHB1dHNUb1NhdmU6IGJvb2xlYW5bXSA9IGdyYWRDb25maWcub3V0cHV0c1RvU2F2ZSB8fCBbXTtcblxuICAgICAgLy8gSWYgc2F2ZUFsbElucHV0cyBpcyB0cnVlLCBhbGwgaW5wdXRzIHdpbGwgYmUgc2F2ZWQuIE90aGVyd2lzZSwgaW5wdXRzXG4gICAgICAvLyBzcGVjaWZpZWQgaW4gaW5wdXRzVG9TYXZlIHdpbGwgYmUgc2F2ZWQuXG4gICAgICBsZXQgaW5wdXRUZW5zb3JzVG9TYXZlOiBUZW5zb3JbXTtcbiAgICAgIGlmIChncmFkQ29uZmlnLnNhdmVBbGxJbnB1dHMpIHtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICBBcnJheS5pc0FycmF5KGlucHV0cyksXG4gICAgICAgICAgICAoKSA9PiAnc2F2ZUFsbElucHV0cyBpcyB0cnVlLCBleHBlY3RlZCBpbnB1dHMgdG8gYmUgYW4gYXJyYXkuJyk7XG5cbiAgICAgICAgaW5wdXRUZW5zb3JzVG9TYXZlID0gT2JqZWN0LmtleXMoaW5wdXRzKS5tYXAoKGtleSkgPT4gaW5wdXRzW2tleV0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgaW5wdXRUZW5zb3JzVG9TYXZlID0gaW5wdXRzVG9TYXZlLm1hcCgoaW5wdXROYW1lKSA9PiBpbnB1dHNbaW5wdXROYW1lXSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IG91dHB1dFRlbnNvcnNUb1NhdmU6IFRlbnNvcltdID1cbiAgICAgICAgICBvdXRwdXRzLmZpbHRlcigoXywgaSkgPT4gb3V0cHV0c1RvU2F2ZVtpXSk7XG5cbiAgICAgIHJldHVybiBpbnB1dFRlbnNvcnNUb1NhdmUuY29uY2F0KG91dHB1dFRlbnNvcnNUb1NhdmUpO1xuICAgIH1cbiAgICAvLyBXZSByZXR1cm4gYW4gZW1wdHkgbGlzdCByYXRoZXIgdGhhbiB0aHJvdyBhbiBlcnJvciBiZWNhdXNlIHRoZSBrZXJuZWwgd2VcbiAgICAvLyBhcmUgbG9va2luZyB1cCBtYXkgbm90IGFjdHVhbGx5IGJlIHJlbGV2YW50IHRvIGJhY2twcm9waW5nIHRocm91Z2ggdGhlXG4gICAgLy8gb3ZlcmFsbCBmdW5jdGlvblxuICAgIC8vXG4gICAgLy8gU2VlICdkb2VzIG5vdCBlcnJvciBpZiBpcnJlbGV2YW50IChwcnVuZWQpIG9wcyBhcmUgbWlzc2luZyBncmFkcycgdGVzdFxuICAgIC8vIGluIGdyYWRpZW50c190ZXN0LnRzIGZvciBhbiBleGFtcGxlLlxuICAgIHJldHVybiBbXTtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbnRlcm5hbCBtZXRob2QgdXNlZCBieSBwdWJsaWMgQVBJcyBmb3IgdGVuc29yIGNyZWF0aW9uLiBNYWtlcyBhIG5ld1xuICAgKiB0ZW5zb3Igd2l0aCB0aGUgcHJvdmlkZWQgc2hhcGUsIGR0eXBlIGFuZCB2YWx1ZXMuIEl0IGFsd2F5c1xuICAgKiBjcmVhdGVzIGEgbmV3IGRhdGEgaWQgYW5kIHdyaXRlcyB0aGUgdmFsdWVzIHRvIHRoZSB1bmRlcmx5aW5nIGJhY2tlbmQuXG4gICAqL1xuICBtYWtlVGVuc29yKFxuICAgICAgdmFsdWVzOiBEYXRhVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGJhY2tlbmQ/OiBLZXJuZWxCYWNrZW5kKTogVGVuc29yIHtcbiAgICBpZiAodmFsdWVzID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignVmFsdWVzIHBhc3NlZCB0byBlbmdpbmUubWFrZVRlbnNvcigpIGFyZSBudWxsJyk7XG4gICAgfVxuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGJhY2tlbmQgPSBiYWNrZW5kIHx8IHRoaXMuYmFja2VuZDtcbiAgICBsZXQgYmFja2VuZFZhbHMgPSB2YWx1ZXMgYXMgQmFja2VuZFZhbHVlcztcbiAgICBpZiAoZHR5cGUgPT09ICdzdHJpbmcnICYmIHV0aWwuaXNTdHJpbmcodmFsdWVzWzBdKSkge1xuICAgICAgYmFja2VuZFZhbHMgPSAodmFsdWVzIGFzIHN0cmluZ1tdKS5tYXAoZCA9PiB1dGlsLmVuY29kZVN0cmluZyhkKSk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGFJZCA9IGJhY2tlbmQud3JpdGUoYmFja2VuZFZhbHMsIHNoYXBlLCBkdHlwZSk7XG4gICAgY29uc3QgdCA9IG5ldyBUZW5zb3Ioc2hhcGUsIGR0eXBlLCBkYXRhSWQsIHRoaXMubmV4dFRlbnNvcklkKCkpO1xuICAgIHRoaXMudHJhY2tUZW5zb3IodCwgYmFja2VuZCk7XG5cbiAgICAvLyBDb3VudCBieXRlcyBmb3Igc3RyaW5nIHRlbnNvcnMuXG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKTtcbiAgICAgIGNvbnN0IG5ld0J5dGVzID0gYnl0ZXNGcm9tU3RyaW5nQXJyYXkoYmFja2VuZFZhbHMgYXMgVWludDhBcnJheVtdKTtcbiAgICAgIHRoaXMuc3RhdGUubnVtQnl0ZXMgKz0gbmV3Qnl0ZXMgLSBpbmZvLmJ5dGVzO1xuICAgICAgaW5mby5ieXRlcyA9IG5ld0J5dGVzO1xuICAgIH1cbiAgICByZXR1cm4gdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbnRlcm5hbCBtZXRob2QgdXNlZCBieSBiYWNrZW5kcy4gTWFrZXMgYSBuZXcgdGVuc29yXG4gICAqIHRoYXQgaXMgYSB3cmFwcGVyIGFyb3VuZCBhbiBleGlzdGluZyBkYXRhIGlkLiBJdCBkb2Vzbid0IGNyZWF0ZVxuICAgKiBhIG5ldyBkYXRhIGlkLCBvbmx5IGluY3JlbWVudHMgdGhlIHJlZiBjb3VudCB1c2VkIGluIG1lbW9yeSB0cmFja2luZy5cbiAgICovXG4gIG1ha2VUZW5zb3JGcm9tRGF0YUlkKFxuICAgICAgZGF0YUlkOiBEYXRhSWQsIHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLFxuICAgICAgYmFja2VuZD86IEtlcm5lbEJhY2tlbmQpOiBUZW5zb3Ige1xuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGNvbnN0IHQgPSBuZXcgVGVuc29yKHNoYXBlLCBkdHlwZSwgZGF0YUlkLCB0aGlzLm5leHRUZW5zb3JJZCgpKTtcbiAgICB0aGlzLnRyYWNrVGVuc29yKHQsIGJhY2tlbmQpO1xuICAgIHJldHVybiB0O1xuICB9XG5cbiAgbWFrZVZhcmlhYmxlKFxuICAgICAgaW5pdGlhbFZhbHVlOiBUZW5zb3IsIHRyYWluYWJsZSA9IHRydWUsIG5hbWU/OiBzdHJpbmcsXG4gICAgICBkdHlwZT86IERhdGFUeXBlKTogVmFyaWFibGUge1xuICAgIG5hbWUgPSBuYW1lIHx8IHRoaXMubmV4dFZhcmlhYmxlSWQoKS50b1N0cmluZygpO1xuICAgIGlmIChkdHlwZSAhPSBudWxsICYmIGR0eXBlICE9PSBpbml0aWFsVmFsdWUuZHR5cGUpIHtcbiAgICAgIGluaXRpYWxWYWx1ZSA9IGluaXRpYWxWYWx1ZS5jYXN0KGR0eXBlKTtcbiAgICB9XG4gICAgY29uc3QgdiA9IG5ldyBWYXJpYWJsZShpbml0aWFsVmFsdWUsIHRyYWluYWJsZSwgbmFtZSwgdGhpcy5uZXh0VGVuc29ySWQoKSk7XG4gICAgaWYgKHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlc1t2Lm5hbWVdICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVmFyaWFibGUgd2l0aCBuYW1lICR7di5uYW1lfSB3YXMgYWxyZWFkeSByZWdpc3RlcmVkYCk7XG4gICAgfVxuICAgIHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlc1t2Lm5hbWVdID0gdjtcbiAgICB0aGlzLmluY1JlZih2LCB0aGlzLmJhY2tlbmQpO1xuICAgIHJldHVybiB2O1xuICB9XG5cbiAgdHJhY2tUZW5zb3IoYTogVGVuc29yLCBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kKTogdm9pZCB7XG4gICAgdGhpcy5zdGF0ZS5udW1UZW5zb3JzKys7XG4gICAgaWYgKGEuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICB0aGlzLnN0YXRlLm51bVN0cmluZ1RlbnNvcnMrKztcbiAgICB9XG4gICAgLy8gQnl0ZXMgZm9yIGNvbXBsZXggbnVtYmVycyBhcmUgY291bnRlZCBieSB0aGVpciBjb21wb25lbnRzLiBCeXRlcyBmb3JcbiAgICAvLyBzdHJpbmcgdGVuc29ycyBhcmUgY291bnRlZCB3aGVuIHdyaXRpbmcgdmFsdWVzLlxuICAgIGxldCBieXRlcyA9IDA7XG4gICAgaWYgKGEuZHR5cGUgIT09ICdjb21wbGV4NjQnICYmIGEuZHR5cGUgIT09ICdzdHJpbmcnKSB7XG4gICAgICBieXRlcyA9IGEuc2l6ZSAqIHV0aWwuYnl0ZXNQZXJFbGVtZW50KGEuZHR5cGUpO1xuICAgIH1cbiAgICB0aGlzLnN0YXRlLm51bUJ5dGVzICs9IGJ5dGVzO1xuXG4gICAgaWYgKCF0aGlzLnN0YXRlLnRlbnNvckluZm8uaGFzKGEuZGF0YUlkKSkge1xuICAgICAgdGhpcy5zdGF0ZS5udW1EYXRhQnVmZmVycysrO1xuICAgICAgdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLnNldChhLmRhdGFJZCwge1xuICAgICAgICBiYWNrZW5kOiBiYWNrZW5kIHx8IHRoaXMuYmFja2VuZCxcbiAgICAgICAgZHR5cGU6IGEuZHR5cGUsXG4gICAgICAgIHNoYXBlOiBhLnNoYXBlLFxuICAgICAgICBieXRlc1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKCEoYSBpbnN0YW5jZW9mIFZhcmlhYmxlKSkge1xuICAgICAgdGhpcy50cmFjayhhKTtcbiAgICB9XG4gIH1cblxuICAvLyBUcmFjayB0aGUgdGVuc29yIGJ5IGRhdGFJZCBhbmQgaW5jcmVhc2UgdGhlIHJlZkNvdW50IGZvciB0aGUgZGF0YUlkIGluIHRoZVxuICAvLyBiYWNrZW5kLlxuICAvLyBUT0RPKHB5dTEwMDU1KTogVGhpcyBpcyBjdXJyZW50bHkgdXNlZCBieSBtYWtlVmFyaWFibGUgbWV0aG9kLCB0byBpbmNyZWFzZVxuICAvLyByZWZDb3VudCBvbiB0aGUgYmFja2VuZCBmb3IgdGhlIGRhdGFJZC4gSXQgY2FuIHBvdGVudGlhbGx5IGJlIHJlcGxhY2VkIHdpdGhcbiAgLy8gSWRlbnRpdHkgb3AgaW5kZWFkIG9mIGNhbGxpbmcgYmFja2VuZCBkaXJlY3RseS5cbiAgaW5jUmVmKGE6IFRlbnNvciwgYmFja2VuZDogS2VybmVsQmFja2VuZCk6IHZvaWQge1xuICAgIHRoaXMudHJhY2tUZW5zb3IoYSwgYmFja2VuZCk7XG4gICAgdGhpcy5iYWNrZW5kLmluY1JlZihhLmRhdGFJZCk7XG4gIH1cblxuICByZW1vdmVEYXRhSWQoZGF0YUlkOiBEYXRhSWQsIGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQpIHtcbiAgICBpZiAodGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmhhcyhkYXRhSWQpICYmXG4gICAgICAgIHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKS5iYWNrZW5kID09PSBiYWNrZW5kKSB7XG4gICAgICB0aGlzLnN0YXRlLnRlbnNvckluZm8uZGVsZXRlKGRhdGFJZCk7XG4gICAgICB0aGlzLnN0YXRlLm51bURhdGFCdWZmZXJzLS07XG4gICAgfVxuICB9XG4gIGRpc3Bvc2VUZW5zb3IoYTogVGVuc29yKTogdm9pZCB7XG4gICAgaWYgKCF0aGlzLnN0YXRlLnRlbnNvckluZm8uaGFzKGEuZGF0YUlkKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBpbmZvID0gdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChhLmRhdGFJZCk7XG5cbiAgICB0aGlzLnN0YXRlLm51bVRlbnNvcnMtLTtcbiAgICBpZiAoYS5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRoaXMuc3RhdGUubnVtU3RyaW5nVGVuc29ycy0tO1xuICAgICAgdGhpcy5zdGF0ZS5udW1CeXRlcyAtPSBpbmZvLmJ5dGVzO1xuICAgIH1cbiAgICAvLyBEb24ndCBjb3VudCBieXRlcyBmb3IgY29tcGxleCBudW1iZXJzIGFzIHRoZXkgYXJlIGNvdW50ZWQgYnkgdGhlaXJcbiAgICAvLyBjb21wb25lbnRzLlxuICAgIGlmIChhLmR0eXBlICE9PSAnY29tcGxleDY0JyAmJiBhLmR0eXBlICE9PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgYnl0ZXMgPSBhLnNpemUgKiB1dGlsLmJ5dGVzUGVyRWxlbWVudChhLmR0eXBlKTtcbiAgICAgIHRoaXMuc3RhdGUubnVtQnl0ZXMgLT0gYnl0ZXM7XG4gICAgfVxuXG4gICAgLy8gUmVtb3ZlIHRoZSByZWZlcmVuY2UgdG8gZGF0YUlkIGlmIGJhY2tlbmQgZGlzcG9zZSB0aGUgZGF0YSBzdWNjZXNzZnVsbHlcbiAgICBpZiAoaW5mby5iYWNrZW5kLmRpc3Bvc2VEYXRhKGEuZGF0YUlkKSkge1xuICAgICAgdGhpcy5yZW1vdmVEYXRhSWQoYS5kYXRhSWQsIGluZm8uYmFja2VuZCk7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhuc3Rob3JhdCk6IENvbnN0cnVjdCBhbiBlcnJvciBhbmQgc2F2ZSB0aGUgc3RhY2sgdHJhY2UgZm9yXG4gICAgLy8gZGVidWdnaW5nIHdoZW4gaW4gZGVidWcgbW9kZS4gQ3JlYXRpbmcgYSBzdGFjayB0cmFjZSBpcyB0b28gZXhwZW5zaXZlXG4gICAgLy8gdG8gZG8gdW5jb25kaXRpb25hbGx5LlxuICB9XG5cbiAgZGlzcG9zZVZhcmlhYmxlcygpOiB2b2lkIHtcbiAgICBmb3IgKGNvbnN0IHZhck5hbWUgaW4gdGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzKSB7XG4gICAgICBjb25zdCB2ID0gdGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzW3Zhck5hbWVdO1xuICAgICAgdGhpcy5kaXNwb3NlVmFyaWFibGUodik7XG4gICAgfVxuICB9XG5cbiAgZGlzcG9zZVZhcmlhYmxlKHY6IFZhcmlhYmxlKTogdm9pZCB7XG4gICAgdGhpcy5kaXNwb3NlVGVuc29yKHYpO1xuICAgIGlmICh0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdi5uYW1lXSAhPSBudWxsKSB7XG4gICAgICBkZWxldGUgdGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzW3YubmFtZV07XG4gICAgfVxuICB9XG5cbiAgbWVtb3J5KCk6IE1lbW9yeUluZm8ge1xuICAgIGNvbnN0IGluZm8gPSB0aGlzLmJhY2tlbmQubWVtb3J5KCkgYXMgTWVtb3J5SW5mbztcbiAgICBpbmZvLm51bVRlbnNvcnMgPSB0aGlzLnN0YXRlLm51bVRlbnNvcnM7XG4gICAgaW5mby5udW1EYXRhQnVmZmVycyA9IHRoaXMuc3RhdGUubnVtRGF0YUJ1ZmZlcnM7XG4gICAgaW5mby5udW1CeXRlcyA9IHRoaXMuc3RhdGUubnVtQnl0ZXM7XG4gICAgaWYgKHRoaXMuc3RhdGUubnVtU3RyaW5nVGVuc29ycyA+IDApIHtcbiAgICAgIGluZm8udW5yZWxpYWJsZSA9IHRydWU7XG4gICAgICBpZiAoaW5mby5yZWFzb25zID09IG51bGwpIHtcbiAgICAgICAgaW5mby5yZWFzb25zID0gW107XG4gICAgICB9XG4gICAgICBpbmZvLnJlYXNvbnMucHVzaChcbiAgICAgICAgICAnTWVtb3J5IHVzYWdlIGJ5IHN0cmluZyB0ZW5zb3JzIGlzIGFwcHJveGltYXRlICcgK1xuICAgICAgICAgICcoMiBieXRlcyBwZXIgY2hhcmFjdGVyKScpO1xuICAgIH1cbiAgICByZXR1cm4gaW5mbztcbiAgfVxuXG4gIGFzeW5jIHByb2ZpbGUocXVlcnk6ICgpID0+IChUZW5zb3JDb250YWluZXIgfCBQcm9taXNlPFRlbnNvckNvbnRhaW5lcj4pKTpcbiAgICAgIFByb21pc2U8UHJvZmlsZUluZm8+IHtcbiAgICB0aGlzLnN0YXRlLnByb2ZpbGluZyA9IHRydWU7XG5cbiAgICBjb25zdCBzdGFydEJ5dGVzID0gdGhpcy5zdGF0ZS5udW1CeXRlcztcbiAgICBjb25zdCBzdGFydE51bVRlbnNvcnMgPSB0aGlzLnN0YXRlLm51bVRlbnNvcnM7XG5cbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUua2VybmVscyA9IFtdO1xuICAgIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5yZXN1bHQgPSBhd2FpdCBxdWVyeSgpO1xuXG4gICAgdGhpcy5zdGF0ZS5wcm9maWxpbmcgPSBmYWxzZTtcblxuICAgIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5wZWFrQnl0ZXMgPSBNYXRoLm1heChcbiAgICAgICAgLi4udGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLmtlcm5lbHMubWFwKGQgPT4gZC50b3RhbEJ5dGVzU25hcHNob3QpKTtcbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUubmV3Qnl0ZXMgPSB0aGlzLnN0YXRlLm51bUJ5dGVzIC0gc3RhcnRCeXRlcztcbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUubmV3VGVuc29ycyA9XG4gICAgICAgIHRoaXMuc3RhdGUubnVtVGVuc29ycyAtIHN0YXJ0TnVtVGVuc29ycztcbiAgICBmb3IgKGNvbnN0IGtlcm5lbCBvZiB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUua2VybmVscykge1xuICAgICAga2VybmVsLmtlcm5lbFRpbWVNcyA9IGF3YWl0IGtlcm5lbC5rZXJuZWxUaW1lTXM7XG4gICAgICBrZXJuZWwuZXh0cmFJbmZvID0gYXdhaXQga2VybmVsLmV4dHJhSW5mbztcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZTtcbiAgfVxuXG4gIGlzVGFwZU9uKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLnN0YXRlLmdyYWRpZW50RGVwdGggPiAwICYmIHRoaXMuc3RhdGUua2VybmVsRGVwdGggPT09IDA7XG4gIH1cblxuICBwcml2YXRlIGFkZFRhcGVOb2RlKFxuICAgICAga2VybmVsTmFtZTogc3RyaW5nLCBpbnB1dHM6IE5hbWVkVGVuc29yTWFwLCBvdXRwdXRzOiBUZW5zb3JbXSxcbiAgICAgIGdyYWRpZW50c0Z1bmM6IEdyYWRGdW5jLCBzYXZlZDogVGVuc29yW10sIGF0dHJzOiBOYW1lZEF0dHJNYXApOiB2b2lkIHtcbiAgICBjb25zdCB0YXBlTm9kZTogVGFwZU5vZGUgPVxuICAgICAgICB7aWQ6IHRoaXMuc3RhdGUubmV4dFRhcGVOb2RlSWQrKywga2VybmVsTmFtZSwgaW5wdXRzLCBvdXRwdXRzLCBzYXZlZH07XG5cbiAgICBjb25zdCBncmFkQ29uZmlnID0gZ2V0R3JhZGllbnQoa2VybmVsTmFtZSk7XG4gICAgaWYgKGdyYWRDb25maWcgIT0gbnVsbCkge1xuICAgICAgZ3JhZGllbnRzRnVuYyA9IGdyYWRDb25maWcuZ3JhZEZ1bmM7XG4gICAgfVxuICAgIGlmIChncmFkaWVudHNGdW5jICE9IG51bGwpIHtcbiAgICAgIHRhcGVOb2RlLmdyYWRpZW50ID0gKGR5czogVGVuc29yW10pID0+IHtcbiAgICAgICAgLy8gVE9ETyhzbWlsa292KTogVG8gb3B0aW1pemUgYmFjay1wcm9wLCBwYXNzIGR5cyB0aGF0IGFyZSBub3QgdXNlZCBpblxuICAgICAgICAvLyB0aGUgYmFja3Byb3AgZ3JhcGggdG8gdGhlIHVzZXIgYXMgbnVsbCBpbnN0ZWFkIG9mIHplcm9zXG4gICAgICAgIGR5cyA9IGR5cy5tYXAoKGR5LCBpKSA9PiB7XG4gICAgICAgICAgaWYgKGR5ID09IG51bGwpIHtcbiAgICAgICAgICAgIGNvbnN0IG91dHB1dCA9IG91dHB1dHNbaV07XG4gICAgICAgICAgICBjb25zdCB2YWxzID0gdXRpbC5tYWtlWmVyb3NUeXBlZEFycmF5KG91dHB1dC5zaXplLCBvdXRwdXQuZHR5cGUpO1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMubWFrZVRlbnNvcih2YWxzLCBvdXRwdXQuc2hhcGUsIG91dHB1dC5kdHlwZSk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHJldHVybiBkeTtcbiAgICAgICAgfSk7XG4gICAgICAgIC8vIEdyYWQgZnVuY3Rpb25zIG9mIG9wcyB3aXRoIHNpbmdsZSBvdXRwdXRzIGV4cGVjdCBhIGR5LCB3aGlsZSBvcHNcbiAgICAgICAgLy8gd2l0aCBtdWx0aXBsZSBvdXRwdXRzIGV4cGVjdCBkeXMgKGFycmF5IG9mIGR5KS5cbiAgICAgICAgcmV0dXJuIGdyYWRpZW50c0Z1bmMoZHlzLmxlbmd0aCA+IDEgPyBkeXMgOiBkeXNbMF0sIHNhdmVkLCBhdHRycyk7XG4gICAgICB9O1xuICAgIH1cbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVRhcGUucHVzaCh0YXBlTm9kZSk7XG4gIH1cblxuICBrZWVwPFQgZXh0ZW5kcyBUZW5zb3I+KHJlc3VsdDogVCk6IFQge1xuICAgIHJlc3VsdC5rZXB0ID0gdHJ1ZTtcbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJpdmF0ZSBzdGFydFRhcGUoKSB7XG4gICAgaWYgKHRoaXMuc3RhdGUuZ3JhZGllbnREZXB0aCA9PT0gMCkge1xuICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVUYXBlID0gW107XG4gICAgfVxuICAgIHRoaXMuc3RhdGUuZ3JhZGllbnREZXB0aCsrO1xuICB9XG5cbiAgcHJpdmF0ZSBlbmRUYXBlKCkge1xuICAgIHRoaXMuc3RhdGUuZ3JhZGllbnREZXB0aC0tO1xuICB9XG5cbiAgLyoqXG4gICAqIFN0YXJ0IGEgc2NvcGUuIFVzZSB0aGlzIHdpdGggZW5kU2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBzdGFydFNjb3BlKG5hbWU/OiBzdHJpbmcpIHtcbiAgICBjb25zdCBzY29wZUluZm86IFNjb3BlU3RhdGUgPSB7XG4gICAgICB0cmFjazogW10sXG4gICAgICBuYW1lOiAndW5uYW1lZCBzY29wZScsXG4gICAgICBpZDogdGhpcy5zdGF0ZS5uZXh0U2NvcGVJZCsrXG4gICAgfTtcbiAgICBpZiAobmFtZSkge1xuICAgICAgc2NvcGVJbmZvLm5hbWUgPSBuYW1lO1xuICAgIH1cbiAgICB0aGlzLnN0YXRlLnNjb3BlU3RhY2sucHVzaChzY29wZUluZm8pO1xuICAgIHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUgPSBzY29wZUluZm87XG4gIH1cblxuICAvKipcbiAgICogRW5kIGEgc2NvcGUuIFVzZSB0aGlzIHdpdGggc3RhcnRTY29wZSgpIHRvIGFjaGlldmUgdGhlIHNhbWUgZnVuY3Rpb25hbGl0eVxuICAgKiBhcyBzY29wZSgpIHdpdGhvdXQgdGhlIG5lZWQgZm9yIGEgZnVuY3Rpb24gY2xvc3VyZS5cbiAgICovXG4gIGVuZFNjb3BlKHJlc3VsdD86IFRlbnNvckNvbnRhaW5lcikge1xuICAgIGNvbnN0IHRlbnNvcnNUb1RyYWNrSW5QYXJlbnQgPSBnZXRUZW5zb3JzSW5Db250YWluZXIocmVzdWx0KTtcbiAgICBjb25zdCB0ZW5zb3JzVG9UcmFja0luUGFyZW50U2V0ID1cbiAgICAgICAgbmV3IFNldCh0ZW5zb3JzVG9UcmFja0luUGFyZW50Lm1hcCh0ID0+IHQuaWQpKTtcblxuICAgIC8vIERpc3Bvc2UgdGhlIGFycmF5cyB0cmFja2VkIGluIHRoaXMgc2NvcGUuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLnRyYWNrLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCB0ZW5zb3IgPSB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLnRyYWNrW2ldO1xuICAgICAgaWYgKCF0ZW5zb3Iua2VwdCAmJiAhdGVuc29yc1RvVHJhY2tJblBhcmVudFNldC5oYXModGVuc29yLmlkKSkge1xuICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IG9sZFNjb3BlID0gdGhpcy5zdGF0ZS5zY29wZVN0YWNrLnBvcCgpO1xuICAgIHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUgPSB0aGlzLnN0YXRlLnNjb3BlU3RhY2subGVuZ3RoID09PSAwID9cbiAgICAgICAgbnVsbCA6XG4gICAgICAgIHRoaXMuc3RhdGUuc2NvcGVTdGFja1t0aGlzLnN0YXRlLnNjb3BlU3RhY2subGVuZ3RoIC0gMV07XG5cbiAgICAvLyBUcmFjayB0aGUgY3VycmVudCByZXN1bHQgaW4gdGhlIHBhcmVudCBzY29wZS5cbiAgICB0ZW5zb3JzVG9UcmFja0luUGFyZW50LmZvckVhY2godGVuc29yID0+IHtcbiAgICAgIC8vIE9ubHkgdHJhY2sgdGhlIHRlbnNvciBpZiB3YXMgYWxsb2NhdGVkIGluIHRoZSBpbm5lciBzY29wZSBhbmQgaXMgbm90XG4gICAgICAvLyBnbG9iYWxseSBrZXB0LlxuICAgICAgaWYgKCF0ZW5zb3Iua2VwdCAmJiB0ZW5zb3Iuc2NvcGVJZCA9PT0gb2xkU2NvcGUuaWQpIHtcbiAgICAgICAgdGhpcy50cmFjayh0ZW5zb3IpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgZ3JhZGllbnRzIG9mIGBmYCB3aXRoIHJlc3BlY3QgdG8gZWFjaCBvZiB0aGUgYHhzYC4gVGhlIGdyYWRpZW50c1xuICAgKiByZXR1cm5lZCBhcmUgb2YgdGhlIHNhbWUgbGVuZ3RoIGFzIGB4c2AsIGJ1dCBzb21lIG1pZ2h0IGJlIG51bGwgaWYgYGZgXG4gICAqIHdhcyBub3QgYSBmdW5jdGlvbiBvZiB0aGF0IGB4YC4gSXQgYWxzbyB0YWtlcyBvcHRpb25hbCBkeSB0byBtdWx0aXBseSB0aGVcbiAgICogZ3JhZGllbnQsIHdoaWNoIGRlZmF1bHRzIHRvIGAxYC5cbiAgICovXG4gIGdyYWRpZW50czxUIGV4dGVuZHMgVGVuc29yPihcbiAgICAgIGY6ICgpID0+IFQsIHhzOiBUZW5zb3JbXSwgZHk/OiBULFxuICAgICAgYWxsb3dOb0dyYWRpZW50cyA9IGZhbHNlKToge3ZhbHVlOiBULCBncmFkczogVGVuc29yW119IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeHMubGVuZ3RoID4gMCwgKCkgPT4gJ2dyYWRpZW50cygpIHJlY2VpdmVkIGFuIGVtcHR5IGxpc3Qgb2YgeHMuJyk7XG4gICAgaWYgKGR5ICE9IG51bGwgJiYgZHkuZHR5cGUgIT09ICdmbG9hdDMyJykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBkeSBtdXN0IGhhdmUgJ2Zsb2F0MzInIGR0eXBlLCBidXQgaGFzICcke2R5LmR0eXBlfSdgKTtcbiAgICB9XG5cbiAgICBjb25zdCB5ID0gdGhpcy5zY29wZWRSdW4oXG4gICAgICAgICgpID0+IHRoaXMuc3RhcnRUYXBlKCksICgpID0+IHRoaXMuZW5kVGFwZSgpLFxuICAgICAgICAoKSA9PiB0aGlzLnRpZHkoJ2ZvcndhcmQnLCBmKSk7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeSBpbnN0YW5jZW9mIFRlbnNvcixcbiAgICAgICAgKCkgPT4gJ1RoZSByZXN1bHQgeSByZXR1cm5lZCBieSBmKCkgbXVzdCBiZSBhIHRlbnNvci4nKTtcbiAgICAvLyBGaWx0ZXIgb3V0IHRoZSBub2RlcyB0aGF0IGRvbid0IGNvbm5lY3QgeCA9PiB5LlxuICAgIGNvbnN0IGZpbHRlcmVkVGFwZSA9IGdldEZpbHRlcmVkTm9kZXNYVG9ZKHRoaXMuc3RhdGUuYWN0aXZlVGFwZSwgeHMsIHkpO1xuICAgIGlmICghYWxsb3dOb0dyYWRpZW50cyAmJiBmaWx0ZXJlZFRhcGUubGVuZ3RoID09PSAwICYmIHhzLmxlbmd0aCA+IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQ2Fubm90IGNvbXB1dGUgZ3JhZGllbnQgb2YgeT1mKHgpIHdpdGggcmVzcGVjdCB0byB4LiBNYWtlIHN1cmUgJyArXG4gICAgICAgICAgJ3RoYXQgdGhlIGYgeW91IHBhc3NlZCBlbmNsb3NlcyBhbGwgb3BlcmF0aW9ucyB0aGF0IGxlYWQgZnJvbSB4ICcgK1xuICAgICAgICAgICd0byB5LicpO1xuICAgIH1cblxuICAgIHJldHVybiB0aGlzLnRpZHkoJ2JhY2t3YXJkJywgKCkgPT4ge1xuICAgICAgY29uc3QgYWNjdW11bGF0ZWRHcmFkaWVudE1hcDoge1t0ZW5zb3JJZDogbnVtYmVyXTogVGVuc29yfSA9IHt9O1xuICAgICAgYWNjdW11bGF0ZWRHcmFkaWVudE1hcFt5LmlkXSA9IChkeSA9PSBudWxsKSA/IG9uZXMoeS5zaGFwZSkgOiBkeTtcblxuICAgICAgLy8gQmFja3Byb3AgZ3JhZGllbnRzIHRocm91Z2ggdGhlIGZpbHRlcmVkIG5vZGVzLlxuICAgICAgYmFja3Byb3BhZ2F0ZUdyYWRpZW50cyhcbiAgICAgICAgICBhY2N1bXVsYXRlZEdyYWRpZW50TWFwLCBmaWx0ZXJlZFRhcGUsXG4gICAgICAgICAgLy8gUGFzcyB0aGUgdGlkeSBmdW5jdGlvbiB0byBhdm9pZCBjaXJjdWxhciBkZXAgd2l0aCBgdGFwZS50c2AuXG4gICAgICAgICAgZiA9PiB0aGlzLnRpZHkoZiBhcyBTY29wZUZuPFRlbnNvcj4pLFxuICAgICAgICAgIC8vIFBhc3MgYW4gYWRkIGZ1bmN0aW9uIHRvIGF2b2lkZSBhIGNpcmN1bGFyIGRlcCB3aXRoIGB0YXBlLnRzYC5cbiAgICAgICAgICBhZGQpO1xuICAgICAgY29uc3QgZ3JhZHMgPSB4cy5tYXAoeCA9PiBhY2N1bXVsYXRlZEdyYWRpZW50TWFwW3guaWRdKTtcblxuICAgICAgaWYgKHRoaXMuc3RhdGUuZ3JhZGllbnREZXB0aCA9PT0gMCkge1xuICAgICAgICAvLyBUaGlzIG1lYW5zIHRoYXQgd2UgYXJlIG5vdCBjb21wdXRpbmcgaGlnaGVyLW9yZGVyIGdyYWRpZW50c1xuICAgICAgICAvLyBhbmQgY2FuIGNsZWFuIHVwIHRoZSB0YXBlLlxuICAgICAgICB0aGlzLnN0YXRlLmFjdGl2ZVRhcGUuZm9yRWFjaChub2RlID0+IHtcbiAgICAgICAgICBmb3IgKGNvbnN0IHRlbnNvciBvZiBub2RlLnNhdmVkKSB7XG4gICAgICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIHRoaXMuc3RhdGUuYWN0aXZlVGFwZSA9IG51bGw7XG4gICAgICB9XG4gICAgICByZXR1cm4ge3ZhbHVlOiB5LCBncmFkc307XG4gICAgfSk7XG4gIH1cblxuICBjdXN0b21HcmFkPFQgZXh0ZW5kcyBUZW5zb3I+KGY6IEN1c3RvbUdyYWRpZW50RnVuYzxUPik6XG4gICAgICAoLi4uYXJnczogQXJyYXk8VGVuc29yfEdyYWRTYXZlRnVuYz4pID0+IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB1dGlsLmlzRnVuY3Rpb24oZiksXG4gICAgICAgICgpID0+ICdUaGUgZiBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSBtdXN0IGJlIGEgZnVuY3Rpb24uJyk7XG4gICAgcmV0dXJuICguLi5pbnB1dHM6IFRlbnNvcltdKTogVCA9PiB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBpbnB1dHMuZXZlcnkodCA9PiB0IGluc3RhbmNlb2YgVGVuc29yKSxcbiAgICAgICAgICAoKSA9PiAnVGhlIGFyZ3MgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikoeDEsIHgyLC4uLikgbXVzdCBhbGwgYmUgJyArXG4gICAgICAgICAgICAgICd0ZW5zb3JzJyk7XG5cbiAgICAgIGxldCByZXM6IHtcbiAgICAgICAgdmFsdWU6IFQsXG4gICAgICAgIGdyYWRGdW5jOiAoZHk6IFQsIHNhdmVkOiBUZW5zb3JbXSkgPT4gVGVuc29yIHwgVGVuc29yW10sXG4gICAgICB9O1xuICAgICAgY29uc3QgaW5wdXRNYXA6IE5hbWVkVGVuc29yTWFwID0ge307XG4gICAgICBpbnB1dHMuZm9yRWFjaCgoaW5wdXQsIGkpID0+IHtcbiAgICAgICAgaW5wdXRNYXBbaV0gPSBpbnB1dDtcbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCBmb3J3YXJkRnVuYzogRm9yd2FyZEZ1bmM8VD4gPSAoXywgc2F2ZSkgPT4ge1xuICAgICAgICByZXMgPSBmKC4uLlsuLi5pbnB1dHMsIHNhdmVdKTtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICByZXMudmFsdWUgaW5zdGFuY2VvZiBUZW5zb3IsXG4gICAgICAgICAgICAoKSA9PiAnVGhlIGZ1bmN0aW9uIGYgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikgbXVzdCByZXR1cm4gYW4gJyArXG4gICAgICAgICAgICAgICAgJ29iamVjdCB3aGVyZSBgb2JqLnZhbHVlYCBpcyBhIHRlbnNvcicpO1xuICAgICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAgIHV0aWwuaXNGdW5jdGlvbihyZXMuZ3JhZEZ1bmMpLFxuICAgICAgICAgICAgKCkgPT4gJ1RoZSBmdW5jdGlvbiBmIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpIG11c3QgcmV0dXJuIGFuICcgK1xuICAgICAgICAgICAgICAgICdvYmplY3Qgd2hlcmUgYG9iai5ncmFkRnVuY2AgaXMgYSBmdW5jdGlvbi4nKTtcbiAgICAgICAgcmV0dXJuIHJlcy52YWx1ZTtcbiAgICAgIH07XG5cbiAgICAgIGNvbnN0IGJhY2t3YXJkc0Z1bmMgPSAoZHk6IFQsIHNhdmVkOiBUZW5zb3JbXSkgPT4ge1xuICAgICAgICBjb25zdCBncmFkUmVzID0gcmVzLmdyYWRGdW5jKGR5LCBzYXZlZCk7XG4gICAgICAgIGNvbnN0IGdyYWRzOiBUZW5zb3JbXSA9IEFycmF5LmlzQXJyYXkoZ3JhZFJlcykgPyBncmFkUmVzIDogW2dyYWRSZXNdO1xuICAgICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAgIGdyYWRzLmxlbmd0aCA9PT0gaW5wdXRzLmxlbmd0aCxcbiAgICAgICAgICAgICgpID0+ICdUaGUgZnVuY3Rpb24gZiBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSBtdXN0IHJldHVybiBhbiAnICtcbiAgICAgICAgICAgICAgICAnb2JqZWN0IHdoZXJlIGBvYmouZ3JhZEZ1bmNgIGlzIGEgZnVuY3Rpb24gdGhhdCByZXR1cm5zICcgK1xuICAgICAgICAgICAgICAgICd0aGUgc2FtZSBudW1iZXIgb2YgdGVuc29ycyBhcyBpbnB1dHMgcGFzc2VkIHRvIGYoLi4uKS4nKTtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICBncmFkcy5ldmVyeSh0ID0+IHQgaW5zdGFuY2VvZiBUZW5zb3IpLFxuICAgICAgICAgICAgKCkgPT4gJ1RoZSBmdW5jdGlvbiBmIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpIG11c3QgcmV0dXJuIGFuICcgK1xuICAgICAgICAgICAgICAgICdvYmplY3Qgd2hlcmUgYG9iai5ncmFkRnVuY2AgaXMgYSBmdW5jdGlvbiB0aGF0IHJldHVybnMgJyArXG4gICAgICAgICAgICAgICAgJ2EgbGlzdCBvZiBvbmx5IHRlbnNvcnMuJyk7XG4gICAgICAgIGNvbnN0IGdyYWRNYXA6IHtba2V5OiBzdHJpbmddOiAoKSA9PiBUZW5zb3J9ID0ge307XG4gICAgICAgIGdyYWRzLmZvckVhY2goKGdyYWQsIGkpID0+IHtcbiAgICAgICAgICBncmFkTWFwW2ldID0gKCkgPT4gZ3JhZDtcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBncmFkTWFwO1xuICAgICAgfTtcblxuICAgICAgcmV0dXJuIHRoaXMucnVuS2VybmVsRnVuYyh7XG4gICAgICAgIGZvcndhcmRGdW5jLFxuICAgICAgICBiYWNrd2FyZHNGdW5jLFxuICAgICAgICBpbnB1dHM6IGlucHV0TWFwLFxuICAgICAgfSk7XG4gICAgfTtcbiAgfVxuXG4gIHJlYWRTeW5jKGRhdGFJZDogRGF0YUlkKTogQmFja2VuZFZhbHVlcyB7XG4gICAgLy8gUm91dGUgdGhlIHJlYWQgdG8gdGhlIGNvcnJlY3QgYmFja2VuZC5cbiAgICBjb25zdCBpbmZvID0gdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChkYXRhSWQpO1xuICAgIHJldHVybiBpbmZvLmJhY2tlbmQucmVhZFN5bmMoZGF0YUlkKTtcbiAgfVxuICByZWFkKGRhdGFJZDogRGF0YUlkKTogUHJvbWlzZTxCYWNrZW5kVmFsdWVzPiB7XG4gICAgLy8gUm91dGUgdGhlIHJlYWQgdG8gdGhlIGNvcnJlY3QgYmFja2VuZC5cbiAgICBjb25zdCBpbmZvID0gdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChkYXRhSWQpO1xuICAgIHJldHVybiBpbmZvLmJhY2tlbmQucmVhZChkYXRhSWQpO1xuICB9XG5cbiAgYXN5bmMgdGltZShxdWVyeTogKCkgPT4gdm9pZCk6IFByb21pc2U8VGltaW5nSW5mbz4ge1xuICAgIGNvbnN0IHN0YXJ0ID0gbm93KCk7XG4gICAgY29uc3QgdGltaW5nSW5mbyA9IGF3YWl0IHRoaXMuYmFja2VuZC50aW1lKHF1ZXJ5KSBhcyBUaW1pbmdJbmZvO1xuICAgIHRpbWluZ0luZm8ud2FsbE1zID0gbm93KCkgLSBzdGFydDtcbiAgICByZXR1cm4gdGltaW5nSW5mbztcbiAgfVxuXG4gIC8qKlxuICAgKiBUcmFja3MgYSBUZW5zb3IgaW4gdGhlIGN1cnJlbnQgc2NvcGUgdG8gYmUgYXV0b21hdGljYWxseSBjbGVhbmVkIHVwXG4gICAqIHdoZW4gdGhlIGN1cnJlbnQgc2NvcGUgZW5kcywgYW5kIHJldHVybnMgdGhlIHZhbHVlLlxuICAgKlxuICAgKiBAcGFyYW0gcmVzdWx0IFRoZSBUZW5zb3IgdG8gdHJhY2sgaW4gdGhlIGN1cnJlbnQgc2NvcGUuXG4gICAqL1xuICBwcml2YXRlIHRyYWNrPFQgZXh0ZW5kcyBUZW5zb3I+KHJlc3VsdDogVCk6IFQge1xuICAgIGlmICh0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlICE9IG51bGwpIHtcbiAgICAgIHJlc3VsdC5zY29wZUlkID0gdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS5pZDtcbiAgICAgIHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUudHJhY2sucHVzaChyZXN1bHQpO1xuICAgIH1cblxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBnZXQgcmVnaXN0ZXJlZFZhcmlhYmxlcygpOiBOYW1lZFZhcmlhYmxlTWFwIHtcbiAgICByZXR1cm4gdGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlc2V0cyB0aGUgZW5naW5lIHN0YXRlLiBSZW1vdmVzIGFsbCBiYWNrZW5kcyBidXQgZG9lcyBub3QgcmVtb3ZlXG4gICAqIHJlZ2lzdGVyZWQgYmFja2VuZCBmYWN0b3JpZXMuXG4gICAqL1xuICByZXNldCgpOiB2b2lkIHtcbiAgICAvLyBNYWtlIGFueSBwZW5kaW5nIHByb21pc2Ugb2Jzb2xldGUuXG4gICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXRJZCsrO1xuXG4gICAgdGhpcy5zdGF0ZS5kaXNwb3NlKCk7XG4gICAgdGhpcy5FTlYucmVzZXQoKTtcbiAgICB0aGlzLnN0YXRlID0gbmV3IEVuZ2luZVN0YXRlKCk7XG5cbiAgICBmb3IgKGNvbnN0IGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnkpIHtcbiAgICAgIHRoaXMuZGlzcG9zZVJlZ2lzdGVyZWRLZXJuZWxzKGJhY2tlbmROYW1lKTtcbiAgICAgIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdLmRpc3Bvc2UoKTtcbiAgICAgIGRlbGV0ZSB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXTtcbiAgICB9XG4gICAgdGhpcy5iYWNrZW5kTmFtZSA9IG51bGw7XG4gICAgdGhpcy5iYWNrZW5kSW5zdGFuY2UgPSBudWxsO1xuICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ID0gbnVsbDtcbiAgfVxufVxuXG5mdW5jdGlvbiBvbmVzKHNoYXBlOiBudW1iZXJbXSk6IFRlbnNvciB7XG4gIGNvbnN0IHZhbHVlcyA9IG1ha2VPbmVzVHlwZWRBcnJheShzaXplRnJvbVNoYXBlKHNoYXBlKSwgJ2Zsb2F0MzInKTtcbiAgcmV0dXJuIEVOR0lORS5tYWtlVGVuc29yKHZhbHVlcywgc2hhcGUsICdmbG9hdDMyJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRPck1ha2VFbmdpbmUoKTogRW5naW5lIHtcbiAgY29uc3QgbnMgPSBnZXRHbG9iYWxOYW1lc3BhY2UoKSBhcyB7fSBhcyB7X3RmZW5naW5lOiBFbmdpbmV9O1xuICBpZiAobnMuX3RmZW5naW5lID09IG51bGwpIHtcbiAgICBjb25zdCBlbnZpcm9ubWVudCA9IG5ldyBFbnZpcm9ubWVudChucyk7XG4gICAgbnMuX3RmZW5naW5lID0gbmV3IEVuZ2luZShlbnZpcm9ubWVudCk7XG4gIH1cbiAgc2V0RW52aXJvbm1lbnRHbG9iYWwobnMuX3RmZW5naW5lLkVOVik7XG5cbiAgLy8gVGVsbCB0aGUgY3VycmVudCB0ZW5zb3IgaW50ZXJmYWNlIHRoYXQgdGhlIGdsb2JhbCBlbmdpbmUgaXMgcmVzcG9uc2libGVcbiAgLy8gZm9yIHRyYWNraW5nLlxuICBzZXRUZW5zb3JUcmFja2VyKCgpID0+IG5zLl90ZmVuZ2luZSk7XG4gIHJldHVybiBucy5fdGZlbmdpbmU7XG59XG5cbmV4cG9ydCBjb25zdCBFTkdJTkUgPSBnZXRPck1ha2VFbmdpbmUoKTtcblxuLyoqXG4gKiBBIGltcGxlbWVudGF0aW9uIG9mIHRoZSBhZGQgb3AgZm9yIHVzZSB3aXRoaW4gZW5naW5lIGFuZCB0YXBlLlxuICpcbiAqIFRoaXMgYWxsb3dzIHVzIHRvIGF2b2lkIGEgY2lyY3VsYXIgZGVwZW5kZW5jeSBiZXR3ZWVuIGFkZC50cyBhbmQgZW5naW5lLlxuICogSXQgaXMgZXhwb3J0ZWQgdG8gYmUgYXZhaWxhYmxlIGluIHRhcGUgdGVzdHMuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhZGQoYTogVGVuc29yLCBiOiBUZW5zb3IpOiBUZW5zb3Ige1xuICAvLyBXZSBkdXBsaWNhdGUgQWRkIGhlcmUgdG8gYXZvaWQgYSBjaXJjdWxhciBkZXBlbmRlbmN5IHdpdGggYWRkLnRzLlxuICBjb25zdCBpbnB1dHMgPSB7YSwgYn07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKEFkZCwgaW5wdXRzIGFzIHt9IGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cbiJdfQ==