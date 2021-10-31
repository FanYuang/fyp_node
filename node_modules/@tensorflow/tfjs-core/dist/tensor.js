/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import { getGlobal } from './global_util';
import { tensorToString } from './tensor_format';
import * as util from './util';
import { computeStrides, toNestedArray } from './util';
/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class TensorBuffer {
    constructor(shape, dtype, values) {
        this.dtype = dtype;
        this.shape = shape.slice();
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            const n = values.length;
            util.assert(n === this.size, () => `Length of values '${n}' does not match the size ` +
                `inferred by the shape '${this.size}'.`);
        }
        if (dtype === 'complex64') {
            throw new Error(`complex64 dtype TensorBuffers are not supported. Please create ` +
                `a TensorBuffer for the real and imaginary parts separately and ` +
                `call tf.complex(real, imag).`);
        }
        this.values = values || util.getArrayFromDType(dtype, this.size);
        this.strides = computeStrides(shape);
    }
    /**
     * Sets a value in the buffer at a given location.
     *
     * @param value The value to set.
     * @param locs  The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    set(value, ...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        util.assert(locs.length === this.rank, () => `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
        const index = this.locToIndex(locs);
        this.values[index] = value;
    }
    /**
     * Returns the value in the buffer at the provided location.
     *
     * @param locs The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    get(...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        let i = 0;
        for (const loc of locs) {
            if (loc < 0 || loc >= this.shape[i]) {
                const msg = `Requested out of range element at ${locs}. ` +
                    `  Buffer shape=${this.shape}`;
                throw new Error(msg);
            }
            i++;
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.values[index];
    }
    locToIndex(locs) {
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    }
    indexToLoc(index) {
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        const locs = new Array(this.shape.length);
        for (let i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Creates an immutable `tf.Tensor` object from the buffer.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    toTensor() {
        return trackerFn().makeTensor(this.values, this.shape, this.dtype);
    }
}
// For tracking tensor creation and disposal.
let trackerFn = null;
// Used by chaining methods to call into ops.
let opHandler = null;
// Used to warn about deprecated methods.
let deprecationWarningFn = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
[deprecationWarningFn];
/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
export function setTensorTracker(fn) {
    trackerFn = fn;
}
/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
export function setOpHandler(handler) {
    opHandler = handler;
}
/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
export function setDeprecationWarningFn(fn) {
    deprecationWarningFn = fn;
}
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * For performance reasons, functions that create tensors do not necessarily
 * perform a copy of the data passed to them (e.g. if the data is passed as a
 * `Float32Array`), and changes to the data will change the tensor. This is not
 * a feature and is not supported. To avoid this behavior, use the tensor before
 * changing the input data or create a copy with `copy = tf.add(yourTensor, 0)`.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class Tensor {
    constructor(shape, dtype, dataId, id) {
        /** Whether this tensor has been globally kept. */
        this.kept = false;
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = util.sizeFromShape(shape);
        this.strides = computeStrides(shape);
        this.dataId = dataId;
        this.id = id;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async buffer() {
        const vals = await this.data();
        return opHandler.buffer(this.shape, this.dtype, vals);
    }
    /**
     * Returns a `tf.TensorBuffer` that holds the underlying data.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    bufferSync() {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * asynchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async array() {
        const vals = await this.data();
        return toNestedArray(this.shape, vals, this.dtype === 'complex64');
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    arraySync() {
        return toNestedArray(this.shape, this.dataSync(), this.dtype === 'complex64');
    }
    /**
     * Asynchronously downloads the values from the `tf.Tensor`. Returns a
     * promise of `TypedArray` that resolves when the computation has finished.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async data() {
        this.throwIfDisposed();
        const data = trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            const bytes = await data;
            try {
                return bytes.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /**
     * Synchronously downloads the values from the `tf.Tensor`. This blocks the
     * UI thread until the values are ready, which can cause performance issues.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dataSync() {
        this.throwIfDisposed();
        const data = trackerFn().readSync(this.dataId);
        if (this.dtype === 'string') {
            try {
                return data.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /** Returns the underlying bytes of the tensor's data. */
    async bytes() {
        this.throwIfDisposed();
        const data = await trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            return data;
        }
        else {
            return new Uint8Array(data.buffer);
        }
    }
    /**
     * Disposes `tf.Tensor` from memory.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dispose() {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    }
    get isDisposed() {
        return this.isDisposedInternal;
    }
    throwIfDisposed() {
        if (this.isDisposed) {
            throw new Error(`Tensor is disposed.`);
        }
    }
    /**
     * Prints the `tf.Tensor`. See `tf.print` for details.
     *
     * @param verbose Whether to print verbose information about the tensor,
     *    including dtype and size.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    print(verbose = false) {
        return opHandler.print(this, verbose);
    }
    /**
     * Returns a copy of the tensor. See `tf.clone` for details.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    clone() {
        this.throwIfDisposed();
        return opHandler.clone(this);
    }
    /**
     * Returns a human-readable description of the tensor. Useful for logging.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    toString(verbose = false) {
        const vals = this.dataSync();
        return tensorToString(vals, this.shape, this.dtype, verbose);
    }
    cast(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    variable(trainable = true, name, dtype) {
        this.throwIfDisposed();
        return trackerFn().makeVariable(this, trainable, name, dtype);
    }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
    value: (instance) => {
        // Implementation note: we should use properties of the object that will be
        // defined before the constructor body has finished executing (methods).
        // This is because when this code is transpiled by babel, babel will call
        // classCallCheck before the constructor body is run.
        // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
        return !!instance && instance.data != null && instance.dataSync != null &&
            instance.throwIfDisposed != null;
    }
});
export function getGlobalTensorClass() {
    // Use getGlobal so that we can augment the Tensor class across package
    // boundaries becase the node resolution alg may result in different modules
    // being returned for this file depending on the path they are loaded from.
    return getGlobal('Tensor', () => {
        return Tensor;
    });
}
// Global side effect. Cache global reference to Tensor class
getGlobalTensorClass();
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class Variable extends Tensor {
    constructor(initialValue, trainable, name, tensorId) {
        super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
        this.trainable = trainable;
        this.name = name;
    }
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    assign(newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                `previous value (${this.dtype}) must match`);
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error(`shape of the new value (${newValue.shape}) and ` +
                `previous value (${this.shape}) must match`);
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().incRef(this, null /* backend */);
    }
    dispose() {
        trackerFn().disposeVariable(this);
        this.isDisposedInternal = true;
    }
}
Object.defineProperty(Variable, Symbol.hasInstance, {
    value: (instance) => {
        return instance instanceof Tensor && instance.assign != null &&
            instance.assign instanceof Function;
    }
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVuc29yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy90ZW5zb3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUN4QyxPQUFPLEVBQUMsY0FBYyxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFL0MsT0FBTyxLQUFLLElBQUksTUFBTSxRQUFRLENBQUM7QUFDL0IsT0FBTyxFQUFDLGNBQWMsRUFBRSxhQUFhLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFXckQ7Ozs7Ozs7R0FPRztBQUNILE1BQU0sT0FBTyxZQUFZO0lBTXZCLFlBQVksS0FBa0IsRUFBUyxLQUFRLEVBQUUsTUFBdUI7UUFBakMsVUFBSyxHQUFMLEtBQUssQ0FBRztRQUM3QyxJQUFJLENBQUMsS0FBSyxHQUFHLEtBQUssQ0FBQyxLQUFLLEVBQWlCLENBQUM7UUFDMUMsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBRXRDLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtZQUNsQixNQUFNLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO1lBQ3hCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxLQUFLLElBQUksQ0FBQyxJQUFJLEVBQ2YsR0FBRyxFQUFFLENBQUMscUJBQXFCLENBQUMsNEJBQTRCO2dCQUNwRCwwQkFBMEIsSUFBSSxDQUFDLElBQUksSUFBSSxDQUFDLENBQUM7U0FDbEQ7UUFDRCxJQUFJLEtBQUssS0FBSyxXQUFXLEVBQUU7WUFDekIsTUFBTSxJQUFJLEtBQUssQ0FDWCxpRUFBaUU7Z0JBQ2pFLGlFQUFpRTtnQkFDakUsOEJBQThCLENBQUMsQ0FBQztTQUNyQztRQUNELElBQUksQ0FBQyxNQUFNLEdBQUcsTUFBTSxJQUFJLElBQUksQ0FBQyxpQkFB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QUFLLFFBQVE7Z0JBQ2pELG1CQUFtQixJQUFJLENBQUMsS0FBSyxjQUFjLENBQUMsQ0FBQztTQUNsRDtRQUNELElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLFFBQVEsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLEtBQUssQ0FBQyxFQUFFO1lBQ2pELE1BQU0sSUFBSSxLQUFLLENBQ1gsMkJBQTJCLFFBQVEsQ0FBQyxLQUFLLFFBQVE7Z0JBQ2pELG1CQUFtQixJQUFJLENBQUMsS0FBSyxjQUFjLENBQUMsQ0FBQztTQUNsRDtRQUNELFNBQVMsRUFBRSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNoQyxJQUFJLENBQUMsTUFBTSxHQUFHLFFBQVEsQ0FBQyxNQUFNLENBQUM7UUFDOUIsU0FBUyxFQUFFLENBQUMsTUFBTSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQUVELE9BQU87UUFDTCxTQUFTLEVBQUUsQ0FBQyxlQUFlLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDbEMsSUFBSSxDQUFDLGtCQUFrQixHQUFHLElBQUksQ0FBQztJQUNqQyxDQUFDO0NBQ0Y7QUFFRCxNQUFNLENBQUMsY0FBYyxDQUFDLFFBQVEsRUFBRSxNQUFNLENBQUMsV0FBVyxFQUFFO0lBQ2xELEtBQUssRUFBRSxDQUFDLFFBQWtCLEVBQUUsRUFBRTtRQUM1QixPQUFPLFFBQVEsWUFBWSxNQUFNLElBQUksUUFBUSxDQUFDLE1BQU0sSUFBSSxJQUFJO1lBQ3hELFFBQVEsQ0FBQyxNQUFNLFlBQVksUUFBUSxDQUFDO0lBQzFDLENBQUM7Q0FDRixDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0R2xvYmFsfSBmcm9tICcuL2dsb2JhbF91dGlsJztcbmltcG9ydCB7dGVuc29yVG9TdHJpbmd9IGZyb20gJy4vdGVuc29yX2Zvcm1hdCc7XG5pbXBvcnQge0FycmF5TWFwLCBCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZSwgRGF0YVR5cGVNYXAsIERhdGFWYWx1ZXMsIE51bWVyaWNEYXRhVHlwZSwgUmFuaywgU2hhcGVNYXAsIFNpbmdsZVZhbHVlTWFwLCBUeXBlZEFycmF5fSBmcm9tICcuL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi91dGlsJztcbmltcG9ydCB7Y29tcHV0ZVN0cmlkZXMsIHRvTmVzdGVkQXJyYXl9IGZyb20gJy4vdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgVGVuc29yRGF0YTxEIGV4dGVuZHMgRGF0YVR5cGU+IHtcbiAgZGF0YUlkPzogRGF0YUlkO1xuICB2YWx1ZXM/OiBEYXRhVHlwZU1hcFtEXTtcbn1cblxuLy8gVGhpcyBpbnRlcmZhY2UgbWltaWNzIEtlcm5lbEJhY2tlbmQgKGluIGJhY2tlbmQudHMpLCB3aGljaCB3b3VsZCBjcmVhdGUgYVxuLy8gY2lyY3VsYXIgZGVwZW5kZW5jeSBpZiBpbXBvcnRlZC5cbmV4cG9ydCBpbnRlcmZhY2UgQmFja2VuZCB7fVxuXG4vKipcbiAqIEEgbXV0YWJsZSBvYmplY3QsIHNpbWlsYXIgdG8gYHRmLlRlbnNvcmAsIHRoYXQgYWxsb3dzIHVzZXJzIHRvIHNldCB2YWx1ZXNcbiAqIGF0IGxvY2F0aW9ucyBiZWZvcmUgY29udmVydGluZyB0byBhbiBpbW11dGFibGUgYHRmLlRlbnNvcmAuXG4gKlxuICogU2VlIGB0Zi5idWZmZXJgIGZvciBjcmVhdGluZyBhIHRlbnNvciBidWZmZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBjbGFzcyBUZW5zb3JCdWZmZXI8UiBleHRlbmRzIFJhbmssIEQgZXh0ZW5kcyBEYXRhVHlwZSA9ICdmbG9hdDMyJz4ge1xuICBzaXplOiBudW1iZXI7XG4gIHNoYXBlOiBTaGFwZU1hcFtSXTtcbiAgc3RyaWRlczogbnVtYmVyW107XG4gIHZhbHVlczogRGF0YVR5cGVNYXBbRF07XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFNoYXBlTWFwW1JdLCBwdWJsaWMgZHR5cGU6IEQsIHZhbHVlcz86IERhdGFUeXBlTWFwW0RdKSB7XG4gICAgdGhpcy5zaGFwZSA9IHNoYXBlLnNsaWNlKCkgYXMgU2hhcGVNYXBbUl07XG4gICAgdGhpcy5zaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgY29uc3QgbiA9IHZhbHVlcy5sZW5ndGg7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBuID09PSB0aGlzLnNpemUsXG4gICAgICAgICAgKCkgPT4gYExlbmd0aCBvZiB2YWx1ZXMgJyR7bn0nIGRvZXMgbm90IG1hdGNoIHRoZSBzaXplIGAgK1xuICAgICAgICAgICAgICBgaW5mZXJyZWQgYnkgdGhlIHNoYXBlICcke3RoaXMuc2l6ZX0nLmApO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYGNvbXBsZXg2NCBkdHlwZSBUZW5zb3JCdWZmZXJzIGFyZSBub3Qgc3VwcG9ydGVkLiBQbGVhc2UgY3JlYXRlIGAgK1xuICAgICAgICAgIGBhIFRlbnNvckJ1ZmZlciBmb3IgdGhlIHJlYWwgYW5kIGltYWdpbmFyeSBwYXJ0cyBzZXBhcmF0ZWx5IGFuZCBgICtcbiAgICAgICAgICBgY2FsbCB0Zi5jb21wbGV4KHJlYWwsIGltYWcpLmApO1xuICAgIH1cbiAgICB0aGlzLnZhbHVlcyA9IHZhbHVlcyB8fCB1dGlsLmdldEFycmF5RnJvbURUeXBlKGR0eXBlLCB0aGlzLnNpemUpO1xuICAgIHRoaXMuc3RyaWRlcyA9IGNvbXB1dGVTdHJpZGVzKHNoYXBlKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXRzIGEgdmFsdWUgaW4gdGhlIGJ1ZmZlciBhdCBhIGdpdmVuIGxvY2F0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gdmFsdWUgVGhlIHZhbHVlIHRvIHNldC5cbiAgICogQHBhcmFtIGxvY3MgIFRoZSBsb2NhdGlvbiBpbmRpY2VzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDcmVhdGlvbid9XG4gICAqL1xuICBzZXQodmFsdWU6IFNpbmdsZVZhbHVlTWFwW0RdLCAuLi5sb2NzOiBudW1iZXJbXSk6IHZvaWQge1xuICAgIGlmIChsb2NzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgbG9jcyA9IFswXTtcbiAgICB9XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGxvY3MubGVuZ3RoID09PSB0aGlzLnJhbmssXG4gICAgICAgICgpID0+IGBUaGUgbnVtYmVyIG9mIHByb3ZpZGVkIGNvb3JkaW5hdGVzICgke2xvY3MubGVuZ3RofSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCB0aGUgcmFuayAoJHt0aGlzLnJhbmt9KWApO1xuXG4gICAgY29uc3QgaW5kZXggPSB0aGlzLmxvY1RvSW5kZXgobG9jcyk7XG4gICAgdGhpcy52YWx1ZXNbaW5kZXhdID0gdmFsdWUgYXMgbnVtYmVyO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIHZhbHVlIGluIHRoZSBidWZmZXIgYXQgdGhlIHByb3ZpZGVkIGxvY2F0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gbG9jcyBUaGUgbG9jYXRpb24gaW5kaWNlcy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICAgKi9cbiAgZ2V0KC4uLmxvY3M6IG51bWJlcltdKTogU2luZ2xlVmFsdWVNYXBbRF0ge1xuICAgIGlmIChsb2NzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgbG9jcyA9IFswXTtcbiAgICB9XG4gICAgbGV0IGkgPSAwO1xuICAgIGZvciAoY29uc3QgbG9jIG9mIGxvY3MpIHtcbiAgICAgIGlmIChsb2MgPCAwIHx8IGxvYyA+PSB0aGlzLnNoYXBlW2ldKSB7XG4gICAgICAgIGNvbnN0IG1zZyA9IGBSZXF1ZXN0ZWQgb3V0IG9mIHJhbmdlIGVsZW1lbnQgYXQgJHtsb2NzfS4gYCArXG4gICAgICAgICAgICBgICBCdWZmZXIgc2hhcGU9JHt0aGlzLnNoYXBlfWA7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihtc2cpO1xuICAgICAgfVxuICAgICAgaSsrO1xuICAgIH1cbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudmFsdWVzW2luZGV4XSBhcyBTaW5nbGVWYWx1ZU1hcFtEXTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jczogbnVtYmVyW10pOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnJhbmsgPT09IDApIHtcbiAgICAgIHJldHVybiAwO1xuICAgIH0gZWxzZSBpZiAodGhpcy5yYW5rID09PSAxKSB7XG4gICAgICByZXR1cm4gbG9jc1swXTtcbiAgICB9XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHJldHVybiBpbmRleDtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IG51bWJlcltdIHtcbiAgICBpZiAodGhpcy5yYW5rID09PSAwKSB7XG4gICAgICByZXR1cm4gW107XG4gICAgfSBlbHNlIGlmICh0aGlzLnJhbmsgPT09IDEpIHtcbiAgICAgIHJldHVybiBbaW5kZXhdO1xuICAgIH1cbiAgICBjb25zdCBsb2NzOiBudW1iZXJbXSA9IG5ldyBBcnJheSh0aGlzLnNoYXBlLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgbG9jc1tpXSA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZXNbaV0pO1xuICAgICAgaW5kZXggLT0gbG9jc1tpXSAqIHRoaXMuc3RyaWRlc1tpXTtcbiAgICB9XG4gICAgbG9jc1tsb2NzLmxlbmd0aCAtIDFdID0gaW5kZXg7XG4gICAgcmV0dXJuIGxvY3M7XG4gIH1cblxuICBnZXQgcmFuaygpIHtcbiAgICByZXR1cm4gdGhpcy5zaGFwZS5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogQ3JlYXRlcyBhbiBpbW11dGFibGUgYHRmLlRlbnNvcmAgb2JqZWN0IGZyb20gdGhlIGJ1ZmZlci5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICAgKi9cbiAgdG9UZW5zb3IoKTogVGVuc29yPFI+IHtcbiAgICByZXR1cm4gdHJhY2tlckZuKCkubWFrZVRlbnNvcih0aGlzLnZhbHVlcywgdGhpcy5zaGFwZSwgdGhpcy5kdHlwZSkgYXNcbiAgICAgICAgVGVuc29yPFI+O1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgVGVuc29yVHJhY2tlciB7XG4gIG1ha2VUZW5zb3IoXG4gICAgICB2YWx1ZXM6IERhdGFWYWx1ZXMsIHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLFxuICAgICAgYmFja2VuZD86IEJhY2tlbmQpOiBUZW5zb3I7XG4gIG1ha2VWYXJpYWJsZShcbiAgICAgIGluaXRpYWxWYWx1ZTogVGVuc29yLCB0cmFpbmFibGU/OiBib29sZWFuLCBuYW1lPzogc3RyaW5nLFxuICAgICAgZHR5cGU/OiBEYXRhVHlwZSk6IFZhcmlhYmxlO1xuICBpbmNSZWYoYTogVGVuc29yLCBiYWNrZW5kOiBCYWNrZW5kKTogdm9pZDtcbiAgZGlzcG9zZVRlbnNvcih0OiBUZW5zb3IpOiB2b2lkO1xuICBkaXNwb3NlVmFyaWFibGUodjogVmFyaWFibGUpOiB2b2lkO1xuICByZWFkKGRhdGFJZDogRGF0YUlkKTogUHJvbWlzZTxCYWNrZW5kVmFsdWVzPjtcbiAgcmVhZFN5bmMoZGF0YUlkOiBEYXRhSWQpOiBCYWNrZW5kVmFsdWVzO1xufVxuXG4vKipcbiAqIFRoZSBUZW5zb3IgY2xhc3MgY2FsbHMgaW50byB0aGlzIGhhbmRsZXIgdG8gZGVsZWdhdGUgY2hhaW5pbmcgb3BlcmF0aW9ucy5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBPcEhhbmRsZXIge1xuICBjYXN0PFQgZXh0ZW5kcyBUZW5zb3I+KHg6IFQsIGR0eXBlOiBEYXRhVHlwZSk6IFQ7XG4gIGJ1ZmZlcjxSIGV4dGVuZHMgUmFuaywgRCBleHRlbmRzIERhdGFUeXBlPihcbiAgICAgIHNoYXBlOiBTaGFwZU1hcFtSXSwgZHR5cGU6IEQsXG4gICAgICB2YWx1ZXM/OiBEYXRhVHlwZU1hcFtEXSk6IFRlbnNvckJ1ZmZlcjxSLCBEPjtcbiAgcHJpbnQ8VCBleHRlbmRzIFRlbnNvcj4oeDogVCwgdmVyYm9zZTogYm9vbGVhbik6IHZvaWQ7XG4gIGNsb25lPFQgZXh0ZW5kcyBUZW5zb3I+KHg6IFQpOiBUO1xuICAvLyBUT0RPKHlhc3NvZ2JhKSBicmluZyByZXNoYXBlIGJhY2s/XG59XG5cbi8vIEZvciB0cmFja2luZyB0ZW5zb3IgY3JlYXRpb24gYW5kIGRpc3Bvc2FsLlxubGV0IHRyYWNrZXJGbjogKCkgPT4gVGVuc29yVHJhY2tlciA9IG51bGw7XG4vLyBVc2VkIGJ5IGNoYWluaW5nIG1ldGhvZHMgdG8gY2FsbCBpbnRvIG9wcy5cbmxldCBvcEhhbmRsZXI6IE9wSGFuZGxlciA9IG51bGw7XG4vLyBVc2VkIHRvIHdhcm4gYWJvdXQgZGVwcmVjYXRlZCBtZXRob2RzLlxubGV0IGRlcHJlY2F0aW9uV2FybmluZ0ZuOiAobXNnOiBzdHJpbmcpID0+IHZvaWQgPSBudWxsO1xuLy8gVGhpcyBoZXJlIHNvIHRoYXQgd2UgY2FuIHVzZSB0aGlzIG1ldGhvZCBvbiBkZXYgYnJhbmNoZXMgYW5kIGtlZXAgdGhlXG4vLyBmdW5jdGlvbmFsaXR5IGF0IG1hc3Rlci5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby11bnVzZWQtZXhwcmVzc2lvblxuW2RlcHJlY2F0aW9uV2FybmluZ0ZuXTtcblxuLyoqXG4gKiBBbiBleHRlcm5hbCBjb25zdW1lciBjYW4gcmVnaXN0ZXIgaXRzZWxmIGFzIHRoZSB0ZW5zb3IgdHJhY2tlci4gVGhpcyB3YXlcbiAqIHRoZSBUZW5zb3IgY2xhc3MgY2FuIG5vdGlmeSB0aGUgdHJhY2tlciBmb3IgZXZlcnkgdGVuc29yIGNyZWF0ZWQgYW5kXG4gKiBkaXNwb3NlZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNldFRlbnNvclRyYWNrZXIoZm46ICgpID0+IFRlbnNvclRyYWNrZXIpIHtcbiAgdHJhY2tlckZuID0gZm47XG59XG5cbi8qKlxuICogQW4gZXh0ZXJuYWwgY29uc3VtZXIgY2FuIHJlZ2lzdGVyIGl0c2VsZiBhcyB0aGUgb3AgaGFuZGxlci4gVGhpcyB3YXkgdGhlXG4gKiBUZW5zb3IgY2xhc3MgY2FuIGhhdmUgY2hhaW5pbmcgbWV0aG9kcyB0aGF0IGNhbGwgaW50byBvcHMgdmlhIHRoZSBvcFxuICogaGFuZGxlci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNldE9wSGFuZGxlcihoYW5kbGVyOiBPcEhhbmRsZXIpIHtcbiAgb3BIYW5kbGVyID0gaGFuZGxlcjtcbn1cblxuLyoqXG4gKiBTZXRzIHRoZSBkZXByZWNhdGlvbiB3YXJuaW5nIGZ1bmN0aW9uIHRvIGJlIHVzZWQgYnkgdGhpcyBmaWxlLiBUaGlzIHdheSB0aGVcbiAqIFRlbnNvciBjbGFzcyBjYW4gYmUgYSBsZWFmIGJ1dCBzdGlsbCB1c2UgdGhlIGVudmlyb25tZW50LlxuICovXG5leHBvcnQgZnVuY3Rpb24gc2V0RGVwcmVjYXRpb25XYXJuaW5nRm4oZm46IChtc2c6IHN0cmluZykgPT4gdm9pZCkge1xuICBkZXByZWNhdGlvbldhcm5pbmdGbiA9IGZuO1xufVxuXG4vKipcbiAqIFdlIHdyYXAgZGF0YSBpZCBzaW5jZSB3ZSB1c2Ugd2VhayBtYXAgdG8gYXZvaWQgbWVtb3J5IGxlYWtzLlxuICogU2luY2Ugd2UgaGF2ZSBvdXIgb3duIG1lbW9yeSBtYW5hZ2VtZW50LCB3ZSBoYXZlIGEgcmVmZXJlbmNlIGNvdW50ZXJcbiAqIG1hcHBpbmcgYSB0ZW5zb3IgdG8gaXRzIGRhdGEsIHNvIHRoZXJlIGlzIGFsd2F5cyBhIHBvaW50ZXIgKGV2ZW4gaWYgdGhhdFxuICogZGF0YSBpcyBvdGhlcndpc2UgZ2FyYmFnZSBjb2xsZWN0YWJsZSkuXG4gKiBTZWUgaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvSmF2YVNjcmlwdC9SZWZlcmVuY2UvXG4gKiBHbG9iYWxfT2JqZWN0cy9XZWFrTWFwXG4gKi9cbmV4cG9ydCB0eXBlIERhdGFJZCA9IG9iamVjdDsgIC8vIG9iamVjdCBpbnN0ZWFkIG9mIHt9IHRvIGZvcmNlIG5vbi1wcmltaXRpdmUuXG5cbi8vIERlY2xhcmUgdGhpcyBuYW1lc3BhY2UgdG8gbWFrZSBUZW5zb3IgY2xhc3MgYXVnbWVudGF0aW9uIHdvcmsgaW4gZ29vZ2xlMy5cbmV4cG9ydCBkZWNsYXJlIG5hbWVzcGFjZSBUZW5zb3Ige31cbi8qKlxuICogQSBgdGYuVGVuc29yYCBvYmplY3QgcmVwcmVzZW50cyBhbiBpbW11dGFibGUsIG11bHRpZGltZW5zaW9uYWwgYXJyYXkgb2ZcbiAqIG51bWJlcnMgdGhhdCBoYXMgYSBzaGFwZSBhbmQgYSBkYXRhIHR5cGUuXG4gKlxuICogRm9yIHBlcmZvcm1hbmNlIHJlYXNvbnMsIGZ1bmN0aW9ucyB0aGF0IGNyZWF0ZSB0ZW5zb3JzIGRvIG5vdCBuZWNlc3NhcmlseVxuICogcGVyZm9ybSBhIGNvcHkgb2YgdGhlIGRhdGEgcGFzc2VkIHRvIHRoZW0gKGUuZy4gaWYgdGhlIGRhdGEgaXMgcGFzc2VkIGFzIGFcbiAqIGBGbG9hdDMyQXJyYXlgKSwgYW5kIGNoYW5nZXMgdG8gdGhlIGRhdGEgd2lsbCBjaGFuZ2UgdGhlIHRlbnNvci4gVGhpcyBpcyBub3RcbiAqIGEgZmVhdHVyZSBhbmQgaXMgbm90IHN1cHBvcnRlZC4gVG8gYXZvaWQgdGhpcyBiZWhhdmlvciwgdXNlIHRoZSB0ZW5zb3IgYmVmb3JlXG4gKiBjaGFuZ2luZyB0aGUgaW5wdXQgZGF0YSBvciBjcmVhdGUgYSBjb3B5IHdpdGggYGNvcHkgPSB0Zi5hZGQoeW91clRlbnNvciwgMClgLlxuICpcbiAqIFNlZSBgdGYudGVuc29yYCBmb3IgZGV0YWlscyBvbiBob3cgdG8gY3JlYXRlIGEgYHRmLlRlbnNvcmAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBjbGFzcyBUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiB7XG4gIC8qKiBVbmlxdWUgaWQgb2YgdGhpcyB0ZW5zb3IuICovXG4gIHJlYWRvbmx5IGlkOiBudW1iZXI7XG4gIC8qKlxuICAgKiBJZCBvZiB0aGUgYnVja2V0IGhvbGRpbmcgdGhlIGRhdGEgZm9yIHRoaXMgdGVuc29yLiBNdWx0aXBsZSBhcnJheXMgY2FuXG4gICAqIHBvaW50IHRvIHRoZSBzYW1lIGJ1Y2tldCAoZS5nLiB3aGVuIGNhbGxpbmcgYXJyYXkucmVzaGFwZSgpKS5cbiAgICovXG4gIGRhdGFJZDogRGF0YUlkO1xuICAvKiogVGhlIHNoYXBlIG9mIHRoZSB0ZW5zb3IuICovXG4gIHJlYWRvbmx5IHNoYXBlOiBTaGFwZU1hcFtSXTtcbiAgLyoqIE51bWJlciBvZiBlbGVtZW50cyBpbiB0aGUgdGVuc29yLiAqL1xuICByZWFkb25seSBzaXplOiBudW1iZXI7XG4gIC8qKiBUaGUgZGF0YSB0eXBlIGZvciB0aGUgYXJyYXkuICovXG4gIHJlYWRvbmx5IGR0eXBlOiBEYXRhVHlwZTtcbiAgLyoqIFRoZSByYW5rIHR5cGUgZm9yIHRoZSBhcnJheSAoc2VlIGBSYW5rYCBlbnVtKS4gKi9cbiAgcmVhZG9ubHkgcmFua1R5cGU6IFI7XG5cbiAgLyoqIFdoZXRoZXIgdGhpcyB0ZW5zb3IgaGFzIGJlZW4gZ2xvYmFsbHkga2VwdC4gKi9cbiAga2VwdCA9IGZhbHNlO1xuICAvKiogVGhlIGlkIG9mIHRoZSBzY29wZSB0aGlzIHRlbnNvciBpcyBiZWluZyB0cmFja2VkIGluLiAqL1xuICBzY29wZUlkOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlbGVtZW50cyB0byBza2lwIGluIGVhY2ggZGltZW5zaW9uIHdoZW4gaW5kZXhpbmcuIFNlZVxuICAgKiBodHRwczovL2RvY3Muc2NpcHkub3JnL2RvYy9udW1weS9yZWZlcmVuY2UvZ2VuZXJhdGVkL1xcXG4gICAqIG51bXB5Lm5kYXJyYXkuc3RyaWRlcy5odG1sXG4gICAqL1xuICByZWFkb25seSBzdHJpZGVzOiBudW1iZXJbXTtcblxuICBjb25zdHJ1Y3RvcihzaGFwZTogU2hhcGVNYXBbUl0sIGR0eXBlOiBEYXRhVHlwZSwgZGF0YUlkOiBEYXRhSWQsIGlkOiBudW1iZXIpIHtcbiAgICB0aGlzLnNoYXBlID0gc2hhcGUuc2xpY2UoKSBhcyBTaGFwZU1hcFtSXTtcbiAgICB0aGlzLmR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIHRoaXMuc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgdGhpcy5zdHJpZGVzID0gY29tcHV0ZVN0cmlkZXMoc2hhcGUpO1xuICAgIHRoaXMuZGF0YUlkID0gZGF0YUlkO1xuICAgIHRoaXMuaWQgPSBpZDtcbiAgICB0aGlzLnJhbmtUeXBlID0gKHRoaXMucmFuayA8IDUgPyB0aGlzLnJhbmsudG9TdHJpbmcoKSA6ICdoaWdoZXInKSBhcyBSO1xuICB9XG5cbiAgZ2V0IHJhbmsoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zaGFwZS5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBhIHByb21pc2Ugb2YgYHRmLlRlbnNvckJ1ZmZlcmAgdGhhdCBob2xkcyB0aGUgdW5kZXJseWluZyBkYXRhLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGJ1ZmZlcjxEIGV4dGVuZHMgRGF0YVR5cGUgPSAnZmxvYXQzMic+KCk6IFByb21pc2U8VGVuc29yQnVmZmVyPFIsIEQ+PiB7XG4gICAgY29uc3QgdmFscyA9IGF3YWl0IHRoaXMuZGF0YTxEPigpO1xuICAgIHJldHVybiBvcEhhbmRsZXIuYnVmZmVyKHRoaXMuc2hhcGUsIHRoaXMuZHR5cGUgYXMgRCwgdmFscyk7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBhIGB0Zi5UZW5zb3JCdWZmZXJgIHRoYXQgaG9sZHMgdGhlIHVuZGVybHlpbmcgZGF0YS5cbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBidWZmZXJTeW5jPEQgZXh0ZW5kcyBEYXRhVHlwZSA9ICdmbG9hdDMyJz4oKTogVGVuc29yQnVmZmVyPFIsIEQ+IHtcbiAgICByZXR1cm4gb3BIYW5kbGVyLmJ1ZmZlcih0aGlzLnNoYXBlLCB0aGlzLmR0eXBlIGFzIEQsIHRoaXMuZGF0YVN5bmMoKSk7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgdGVuc29yIGRhdGEgYXMgYSBuZXN0ZWQgYXJyYXkuIFRoZSB0cmFuc2ZlciBvZiBkYXRhIGlzIGRvbmVcbiAgICogYXN5bmNocm9ub3VzbHkuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXN5bmMgYXJyYXkoKTogUHJvbWlzZTxBcnJheU1hcFtSXT4ge1xuICAgIGNvbnN0IHZhbHMgPSBhd2FpdCB0aGlzLmRhdGEoKTtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheSh0aGlzLnNoYXBlLCB2YWxzLCB0aGlzLmR0eXBlID09PSAnY29tcGxleDY0JykgYXNcbiAgICAgICAgQXJyYXlNYXBbUl07XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgdGVuc29yIGRhdGEgYXMgYSBuZXN0ZWQgYXJyYXkuIFRoZSB0cmFuc2ZlciBvZiBkYXRhIGlzIGRvbmVcbiAgICogc3luY2hyb25vdXNseS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhcnJheVN5bmMoKTogQXJyYXlNYXBbUl0ge1xuICAgIHJldHVybiB0b05lc3RlZEFycmF5KFxuICAgICAgICAgICAgICAgdGhpcy5zaGFwZSwgdGhpcy5kYXRhU3luYygpLCB0aGlzLmR0eXBlID09PSAnY29tcGxleDY0JykgYXNcbiAgICAgICAgQXJyYXlNYXBbUl07XG4gIH1cblxuICAvKipcbiAgICogQXN5bmNocm9ub3VzbHkgZG93bmxvYWRzIHRoZSB2YWx1ZXMgZnJvbSB0aGUgYHRmLlRlbnNvcmAuIFJldHVybnMgYVxuICAgKiBwcm9taXNlIG9mIGBUeXBlZEFycmF5YCB0aGF0IHJlc29sdmVzIHdoZW4gdGhlIGNvbXB1dGF0aW9uIGhhcyBmaW5pc2hlZC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyBkYXRhPEQgZXh0ZW5kcyBEYXRhVHlwZSA9IE51bWVyaWNEYXRhVHlwZT4oKTogUHJvbWlzZTxEYXRhVHlwZU1hcFtEXT4ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZGF0YSA9IHRyYWNrZXJGbigpLnJlYWQodGhpcy5kYXRhSWQpO1xuICAgIGlmICh0aGlzLmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgYnl0ZXMgPSBhd2FpdCBkYXRhIGFzIFVpbnQ4QXJyYXlbXTtcbiAgICAgIHRyeSB7XG4gICAgICAgIHJldHVybiBieXRlcy5tYXAoYiA9PiB1dGlsLmRlY29kZVN0cmluZyhiKSkgYXMgRGF0YVR5cGVNYXBbRF07XG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0ZhaWxlZCB0byBkZWNvZGUgdGhlIHN0cmluZyBieXRlcyBpbnRvIHV0Zi04LiAnICtcbiAgICAgICAgICAgICdUbyBnZXQgdGhlIG9yaWdpbmFsIGJ5dGVzLCBjYWxsIHRlbnNvci5ieXRlcygpLicpO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZGF0YSBhcyBQcm9taXNlPERhdGFUeXBlTWFwW0RdPjtcbiAgfVxuXG4gIC8qKlxuICAgKiBTeW5jaHJvbm91c2x5IGRvd25sb2FkcyB0aGUgdmFsdWVzIGZyb20gdGhlIGB0Zi5UZW5zb3JgLiBUaGlzIGJsb2NrcyB0aGVcbiAgICogVUkgdGhyZWFkIHVudGlsIHRoZSB2YWx1ZXMgYXJlIHJlYWR5LCB3aGljaCBjYW4gY2F1c2UgcGVyZm9ybWFuY2UgaXNzdWVzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGRhdGFTeW5jPEQgZXh0ZW5kcyBEYXRhVHlwZSA9IE51bWVyaWNEYXRhVHlwZT4oKTogRGF0YVR5cGVNYXBbRF0ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZGF0YSA9IHRyYWNrZXJGbigpLnJlYWRTeW5jKHRoaXMuZGF0YUlkKTtcbiAgICBpZiAodGhpcy5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIHJldHVybiAoZGF0YSBhcyBVaW50OEFycmF5W10pLm1hcChiID0+IHV0aWwuZGVjb2RlU3RyaW5nKGIpKSBhc1xuICAgICAgICAgICAgRGF0YVR5cGVNYXBbRF07XG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0ZhaWxlZCB0byBkZWNvZGUgdGhlIHN0cmluZyBieXRlcyBpbnRvIHV0Zi04LiAnICtcbiAgICAgICAgICAgICdUbyBnZXQgdGhlIG9yaWdpbmFsIGJ5dGVzLCBjYWxsIHRlbnNvci5ieXRlcygpLicpO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZGF0YSBhcyBEYXRhVHlwZU1hcFtEXTtcbiAgfVxuXG4gIC8qKiBSZXR1cm5zIHRoZSB1bmRlcmx5aW5nIGJ5dGVzIG9mIHRoZSB0ZW5zb3IncyBkYXRhLiAqL1xuICBhc3luYyBieXRlcygpOiBQcm9taXNlPFVpbnQ4QXJyYXlbXXxVaW50OEFycmF5PiB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBjb25zdCBkYXRhID0gYXdhaXQgdHJhY2tlckZuKCkucmVhZCh0aGlzLmRhdGFJZCk7XG4gICAgaWYgKHRoaXMuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICByZXR1cm4gZGF0YSBhcyBVaW50OEFycmF5W107XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBuZXcgVWludDhBcnJheSgoZGF0YSBhcyBUeXBlZEFycmF5KS5idWZmZXIpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBEaXNwb3NlcyBgdGYuVGVuc29yYCBmcm9tIG1lbW9yeS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBkaXNwb3NlKCk6IHZvaWQge1xuICAgIGlmICh0aGlzLmlzRGlzcG9zZWQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgdHJhY2tlckZuKCkuZGlzcG9zZVRlbnNvcih0aGlzKTtcbiAgICB0aGlzLmlzRGlzcG9zZWRJbnRlcm5hbCA9IHRydWU7XG4gIH1cblxuICBwcm90ZWN0ZWQgaXNEaXNwb3NlZEludGVybmFsID0gZmFsc2U7XG4gIGdldCBpc0Rpc3Bvc2VkKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLmlzRGlzcG9zZWRJbnRlcm5hbDtcbiAgfVxuXG4gIHRocm93SWZEaXNwb3NlZCgpIHtcbiAgICBpZiAodGhpcy5pc0Rpc3Bvc2VkKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFRlbnNvciBpcyBkaXNwb3NlZC5gKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogUHJpbnRzIHRoZSBgdGYuVGVuc29yYC4gU2VlIGB0Zi5wcmludGAgZm9yIGRldGFpbHMuXG4gICAqXG4gICAqIEBwYXJhbSB2ZXJib3NlIFdoZXRoZXIgdG8gcHJpbnQgdmVyYm9zZSBpbmZvcm1hdGlvbiBhYm91dCB0aGUgdGVuc29yLFxuICAgKiAgICBpbmNsdWRpbmcgZHR5cGUgYW5kIHNpemUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgcHJpbnQodmVyYm9zZSA9IGZhbHNlKTogdm9pZCB7XG4gICAgcmV0dXJuIG9wSGFuZGxlci5wcmludCh0aGlzLCB2ZXJib3NlKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIGEgY29weSBvZiB0aGUgdGVuc29yLiBTZWUgYHRmLmNsb25lYCBmb3IgZGV0YWlscy5cbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBjbG9uZTxUIGV4dGVuZHMgVGVuc29yPih0aGlzOiBUKTogVCB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gb3BIYW5kbGVyLmNsb25lKHRoaXMpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBodW1hbi1yZWFkYWJsZSBkZXNjcmlwdGlvbiBvZiB0aGUgdGVuc29yLiBVc2VmdWwgZm9yIGxvZ2dpbmcuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgdG9TdHJpbmcodmVyYm9zZSA9IGZhbHNlKTogc3RyaW5nIHtcbiAgICBjb25zdCB2YWxzID0gdGhpcy5kYXRhU3luYygpO1xuICAgIHJldHVybiB0ZW5zb3JUb1N0cmluZyh2YWxzLCB0aGlzLnNoYXBlLCB0aGlzLmR0eXBlLCB2ZXJib3NlKTtcbiAgfVxuXG4gIGNhc3Q8VCBleHRlbmRzIHRoaXM+KGR0eXBlOiBEYXRhVHlwZSk6IFQge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIG9wSGFuZGxlci5jYXN0KHRoaXMgYXMgVCwgZHR5cGUpO1xuICB9XG4gIHZhcmlhYmxlKHRyYWluYWJsZSA9IHRydWUsIG5hbWU/OiBzdHJpbmcsIGR0eXBlPzogRGF0YVR5cGUpOiBWYXJpYWJsZTxSPiB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gdHJhY2tlckZuKCkubWFrZVZhcmlhYmxlKHRoaXMsIHRyYWluYWJsZSwgbmFtZSwgZHR5cGUpIGFzXG4gICAgICAgIFZhcmlhYmxlPFI+O1xuICB9XG59XG5PYmplY3QuZGVmaW5lUHJvcGVydHkoVGVuc29yLCBTeW1ib2wuaGFzSW5zdGFuY2UsIHtcbiAgdmFsdWU6IChpbnN0YW5jZTogVGVuc29yKSA9PiB7XG4gICAgLy8gSW1wbGVtZW50YXRpb24gbm90ZTogd2Ugc2hvdWxkIHVzZSBwcm9wZXJ0aWVzIG9mIHRoZSBvYmplY3QgdGhhdCB3aWxsIGJlXG4gICAgLy8gZGVmaW5lZCBiZWZvcmUgdGhlIGNvbnN0cnVjdG9yIGJvZHkgaGFzIGZpbmlzaGVkIGV4ZWN1dGluZyAobWV0aG9kcykuXG4gICAgLy8gVGhpcyBpcyBiZWNhdXNlIHdoZW4gdGhpcyBjb2RlIGlzIHRyYW5zcGlsZWQgYnkgYmFiZWwsIGJhYmVsIHdpbGwgY2FsbFxuICAgIC8vIGNsYXNzQ2FsbENoZWNrIGJlZm9yZSB0aGUgY29uc3RydWN0b3IgYm9keSBpcyBydW4uXG4gICAgLy8gU2VlIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzMzODQgZm9yIGJhY2tzdG9yeS5cbiAgICByZXR1cm4gISFpbnN0YW5jZSAmJiBpbnN0YW5jZS5kYXRhICE9IG51bGwgJiYgaW5zdGFuY2UuZGF0YVN5bmMgIT0gbnVsbCAmJlxuICAgICAgICBpbnN0YW5jZS50aHJvd0lmRGlzcG9zZWQgIT0gbnVsbDtcbiAgfVxufSk7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRHbG9iYWxUZW5zb3JDbGFzcygpIHtcbiAgLy8gVXNlIGdldEdsb2JhbCBzbyB0aGF0IHdlIGNhbiBhdWdtZW50IHRoZSBUZW5zb3IgY2xhc3MgYWNyb3NzIHBhY2thZ2VcbiAgLy8gYm91bmRhcmllcyBiZWNhc2UgdGhlIG5vZGUgcmVzb2x1dGlvbiBhbGcgbWF5IHJlc3VsdCBpbiBkaWZmZXJlbnQgbW9kdWxlc1xuICAvLyBiZWluZyByZXR1cm5lZCBmb3IgdGhpcyBmaWxlIGRlcGVuZGluZyBvbiB0aGUgcGF0aCB0aGV5IGFyZSBsb2FkZWQgZnJvbS5cbiAgcmV0dXJuIGdldEdsb2JhbCgnVGVuc29yJywgKCkgPT4ge1xuICAgIHJldHVybiBUZW5zb3I7XG4gIH0pO1xufVxuXG4vLyBHbG9iYWwgc2lkZSBlZmZlY3QuIENhY2hlIGdsb2JhbCByZWZlcmVuY2UgdG8gVGVuc29yIGNsYXNzXG5nZXRHbG9iYWxUZW5zb3JDbGFzcygpO1xuXG5leHBvcnQgaW50ZXJmYWNlIE51bWVyaWNUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiBleHRlbmRzIFRlbnNvcjxSPiB7XG4gIGR0eXBlOiBOdW1lcmljRGF0YVR5cGU7XG4gIGRhdGFTeW5jPEQgZXh0ZW5kcyBEYXRhVHlwZSA9IE51bWVyaWNEYXRhVHlwZT4oKTogRGF0YVR5cGVNYXBbRF07XG4gIGRhdGE8RCBleHRlbmRzIERhdGFUeXBlID0gTnVtZXJpY0RhdGFUeXBlPigpOiBQcm9taXNlPERhdGFUeXBlTWFwW0RdPjtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBTdHJpbmdUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiBleHRlbmRzIFRlbnNvcjxSPiB7XG4gIGR0eXBlOiAnc3RyaW5nJztcbiAgZGF0YVN5bmM8RCBleHRlbmRzIERhdGFUeXBlID0gJ3N0cmluZyc+KCk6IERhdGFUeXBlTWFwW0RdO1xuICBkYXRhPEQgZXh0ZW5kcyBEYXRhVHlwZSA9ICdzdHJpbmcnPigpOiBQcm9taXNlPERhdGFUeXBlTWFwW0RdPjtcbn1cblxuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgU2NhbGFyID0gVGVuc29yPFJhbmsuUjA+O1xuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgVGVuc29yMUQgPSBUZW5zb3I8UmFuay5SMT47XG4vKiogQGRvY2xpbmsgVGVuc29yICovXG5leHBvcnQgdHlwZSBUZW5zb3IyRCA9IFRlbnNvcjxSYW5rLlIyPjtcbi8qKiBAZG9jbGluayBUZW5zb3IgKi9cbmV4cG9ydCB0eXBlIFRlbnNvcjNEID0gVGVuc29yPFJhbmsuUjM+O1xuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgVGVuc29yNEQgPSBUZW5zb3I8UmFuay5SND47XG4vKiogQGRvY2xpbmsgVGVuc29yICovXG5leHBvcnQgdHlwZSBUZW5zb3I1RCA9IFRlbnNvcjxSYW5rLlI1Pjtcbi8qKiBAZG9jbGluayBUZW5zb3IgKi9cbmV4cG9ydCB0eXBlIFRlbnNvcjZEID0gVGVuc29yPFJhbmsuUjY+O1xuXG4vKipcbiAqIEEgbXV0YWJsZSBgdGYuVGVuc29yYCwgdXNlZnVsIGZvciBwZXJzaXN0aW5nIHN0YXRlLCBlLmcuIGZvciB0cmFpbmluZy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAqL1xuZXhwb3J0IGNsYXNzIFZhcmlhYmxlPFIgZXh0ZW5kcyBSYW5rID0gUmFuaz4gZXh0ZW5kcyBUZW5zb3I8Uj4ge1xuICBuYW1lOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBpbml0aWFsVmFsdWU6IFRlbnNvcjxSPiwgcHVibGljIHRyYWluYWJsZTogYm9vbGVhbiwgbmFtZTogc3RyaW5nLFxuICAgICAgdGVuc29ySWQ6IG51bWJlcikge1xuICAgIHN1cGVyKFxuICAgICAgICBpbml0aWFsVmFsdWUuc2hhcGUsIGluaXRpYWxWYWx1ZS5kdHlwZSwgaW5pdGlhbFZhbHVlLmRhdGFJZCwgdGVuc29ySWQpO1xuICAgIHRoaXMubmFtZSA9IG5hbWU7XG4gIH1cblxuICAvKipcbiAgICogQXNzaWduIGEgbmV3IGB0Zi5UZW5zb3JgIHRvIHRoaXMgdmFyaWFibGUuIFRoZSBuZXcgYHRmLlRlbnNvcmAgbXVzdCBoYXZlXG4gICAqIHRoZSBzYW1lIHNoYXBlIGFuZCBkdHlwZSBhcyB0aGUgb2xkIGB0Zi5UZW5zb3JgLlxuICAgKlxuICAgKiBAcGFyYW0gbmV3VmFsdWUgTmV3IHRlbnNvciB0byBiZSBhc3NpZ25lZCB0byB0aGlzIHZhcmlhYmxlLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzc2lnbihuZXdWYWx1ZTogVGVuc29yPFI+KTogdm9pZCB7XG4gICAgaWYgKG5ld1ZhbHVlLmR0eXBlICE9PSB0aGlzLmR0eXBlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYGR0eXBlIG9mIHRoZSBuZXcgdmFsdWUgKCR7bmV3VmFsdWUuZHR5cGV9KSBhbmQgYCArXG4gICAgICAgICAgYHByZXZpb3VzIHZhbHVlICgke3RoaXMuZHR5cGV9KSBtdXN0IG1hdGNoYCk7XG4gICAgfVxuICAgIGlmICghdXRpbC5hcnJheXNFcXVhbChuZXdWYWx1ZS5zaGFwZSwgdGhpcy5zaGFwZSkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgc2hhcGUgb2YgdGhlIG5ldyB2YWx1ZSAoJHtuZXdWYWx1ZS5zaGFwZX0pIGFuZCBgICtcbiAgICAgICAgICBgcHJldmlvdXMgdmFsdWUgKCR7dGhpcy5zaGFwZX0pIG11c3QgbWF0Y2hgKTtcbiAgICB9XG4gICAgdHJhY2tlckZuKCkuZGlzcG9zZVRlbnNvcih0aGlzKTtcbiAgICB0aGlzLmRhdGFJZCA9IG5ld1ZhbHVlLmRhdGFJZDtcbiAgICB0cmFja2VyRm4oKS5pbmNSZWYodGhpcywgbnVsbCAvKiBiYWNrZW5kICovKTtcbiAgfVxuXG4gIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgdHJhY2tlckZuKCkuZGlzcG9zZVZhcmlhYmxlKHRoaXMpO1xuICAgIHRoaXMuaXNEaXNwb3NlZEludGVybmFsID0gdHJ1ZTtcbiAgfVxufVxuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoVmFyaWFibGUsIFN5bWJvbC5oYXNJbnN0YW5jZSwge1xuICB2YWx1ZTogKGluc3RhbmNlOiBWYXJpYWJsZSkgPT4ge1xuICAgIHJldHVybiBpbnN0YW5jZSBpbnN0YW5jZW9mIFRlbnNvciAmJiBpbnN0YW5jZS5hc3NpZ24gIT0gbnVsbCAmJlxuICAgICAgICBpbnN0YW5jZS5hc3NpZ24gaW5zdGFuY2VvZiBGdW5jdGlvbjtcbiAgfVxufSk7XG4iXX0=