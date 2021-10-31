/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { Exp } from '@tensorflow/tfjs-core';
import { unaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { expImplCPU } from '../kernel_utils/shared';
export const EXP = `return exp(x);`;
export const exp = unaryKernelFunc({
    opSnippet: EXP,
    packedOpSnippet: EXP,
    cpuKernelImpl: expImplCPU,
    dtype: 'float32',
});
export const expConfig = {
    kernelName: Exp,
    backendName: 'webgl',
    kernelFunc: exp
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRXhwLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL0V4cC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUEyQixNQUFNLHVCQUF1QixDQUFDO0FBRXBFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNuRSxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFbEQsTUFBTSxDQUFDLE1BQU0sR0FBRyxHQUFHLGdCQUFnQixDQUFDO0FBQ3BDLE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUM7SUFDakMsU0FBUyxFQUFFLEdBQUc7SUFDZCxlQUFlLEVBQUUsR0FBRztJQUNwQixhQUFhLEVBQUUsVUFBVTtJQUN6QixLQUFLLEVBQUUsU0FBUztDQUNqQixDQUFDLENBQUM7QUFFSCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQWlCO0lBQ3JDLFVBQVUsRUFBRSxHQUFHO0lBQ2YsV0FBVyxFQUFFLE9BQU87SUFDcEIsVUFBVSxFQUFFLEdBQXVCO0NBQ3BDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RXhwLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmN9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7dW5hcnlLZXJuZWxGdW5jfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMva2VybmVsX2Z1bmNzX3V0aWxzJztcbmltcG9ydCB7ZXhwSW1wbENQVX0gZnJvbSAnLi4va2VybmVsX3V0aWxzL3NoYXJlZCc7XG5cbmV4cG9ydCBjb25zdCBFWFAgPSBgcmV0dXJuIGV4cCh4KTtgO1xuZXhwb3J0IGNvbnN0IGV4cCA9IHVuYXJ5S2VybmVsRnVuYyh7XG4gIG9wU25pcHBldDogRVhQLFxuICBwYWNrZWRPcFNuaXBwZXQ6IEVYUCxcbiAgY3B1S2VybmVsSW1wbDogZXhwSW1wbENQVSxcbiAgZHR5cGU6ICdmbG9hdDMyJyxcbn0pO1xuXG5leHBvcnQgY29uc3QgZXhwQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IEV4cCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJnbCcsXG4gIGtlcm5lbEZ1bmM6IGV4cCBhcyB7fSBhcyBLZXJuZWxGdW5jXG59O1xuIl19