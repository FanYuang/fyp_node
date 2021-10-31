/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
// Register the CPU backend as a default backend for tests.
import '@tensorflow/tfjs-backend-cpu';
/**
 * This file is necessary so we register all test environments before we start
 * executing tests.
 */
import { setTestEnvs, setupTestFilters } from './jasmine_util';
// Register all chained ops for tests.
import './public/chained_ops/register_all_chained_ops';
// Register all gradients for tests
import './register_all_gradients';
// Set up a CPU test env as the default test env
setTestEnvs([{ name: 'cpu', backendName: 'cpu', isDataSync: true }]);
const TEST_FILTERS = [];
const customInclude = () => true;
setupTestFilters(TEST_FILTERS, customInclude);
// Import and run all the tests.
// This import, which registers all tests, must be a require because it must run
// after the test environment is set up.
// tslint:disable-next-line:no-require-imports
require('./tests');
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2V0dXBfdGVzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvc2V0dXBfdGVzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCwyREFBMkQ7QUFDM0QsT0FBTyw4QkFBOEIsQ0FBQztBQUN0Qzs7O0dBR0c7QUFDSCxPQUFPLEVBQUMsV0FBVyxFQUFFLGdCQUFnQixFQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFDekUsc0NBQXNDO0FBQ3RDLE9BQU8sK0NBQStDLENBQUM7QUFDdkQsbUNBQW1DO0FBQ25DLE9BQU8sMEJBQTBCLENBQUM7QUFFbEMsZ0RBQWdEO0FBQ2hELFdBQVcsQ0FBQyxDQUFDLEVBQUMsSUFBSSxFQUFFLEtBQUssRUFBRSxXQUFXLEVBQUUsS0FBSyxFQUFFLFVBQVUsRUFBRSxJQUFJLEVBQUMsQ0FBQyxDQUFDLENBQUM7QUFFbkUsTUFBTSxZQUFZLEdBQWlCLEVBQUUsQ0FBQztBQUN0QyxNQUFNLGFBQWEsR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUM7QUFDakMsZ0JBQWdCLENBQUMsWUFBWSxFQUFFLGFBQWEsQ0FBQyxDQUFDO0FBRTlDLGdDQUFnQztBQUNoQyxnRkFBZ0Y7QUFDaEYsd0NBQXdDO0FBQ3hDLDhDQUE4QztBQUM5QyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8vIFJlZ2lzdGVyIHRoZSBDUFUgYmFja2VuZCBhcyBhIGRlZmF1bHQgYmFja2VuZCBmb3IgdGVzdHMuXG5pbXBvcnQgJ0B0ZW5zb3JmbG93L3RmanMtYmFja2VuZC1jcHUnO1xuLyoqXG4gKiBUaGlzIGZpbGUgaXMgbmVjZXNzYXJ5IHNvIHdlIHJlZ2lzdGVyIGFsbCB0ZXN0IGVudmlyb25tZW50cyBiZWZvcmUgd2Ugc3RhcnRcbiAqIGV4ZWN1dGluZyB0ZXN0cy5cbiAqL1xuaW1wb3J0IHtzZXRUZXN0RW52cywgc2V0dXBUZXN0RmlsdGVycywgVGVzdEZpbHRlcn0gZnJvbSAnLi9qYXNtaW5lX3V0aWwnO1xuLy8gUmVnaXN0ZXIgYWxsIGNoYWluZWQgb3BzIGZvciB0ZXN0cy5cbmltcG9ydCAnLi9wdWJsaWMvY2hhaW5lZF9vcHMvcmVnaXN0ZXJfYWxsX2NoYWluZWRfb3BzJztcbi8vIFJlZ2lzdGVyIGFsbCBncmFkaWVudHMgZm9yIHRlc3RzXG5pbXBvcnQgJy4vcmVnaXN0ZXJfYWxsX2dyYWRpZW50cyc7XG5cbi8vIFNldCB1cCBhIENQVSB0ZXN0IGVudiBhcyB0aGUgZGVmYXVsdCB0ZXN0IGVudlxuc2V0VGVzdEVudnMoW3tuYW1lOiAnY3B1JywgYmFja2VuZE5hbWU6ICdjcHUnLCBpc0RhdGFTeW5jOiB0cnVlfV0pO1xuXG5jb25zdCBURVNUX0ZJTFRFUlM6IFRlc3RGaWx0ZXJbXSA9IFtdO1xuY29uc3QgY3VzdG9tSW5jbHVkZSA9ICgpID0+IHRydWU7XG5zZXR1cFRlc3RGaWx0ZXJzKFRFU1RfRklMVEVSUywgY3VzdG9tSW5jbHVkZSk7XG5cbi8vIEltcG9ydCBhbmQgcnVuIGFsbCB0aGUgdGVzdHMuXG4vLyBUaGlzIGltcG9ydCwgd2hpY2ggcmVnaXN0ZXJzIGFsbCB0ZXN0cywgbXVzdCBiZSBhIHJlcXVpcmUgYmVjYXVzZSBpdCBtdXN0IHJ1blxuLy8gYWZ0ZXIgdGhlIHRlc3QgZW52aXJvbm1lbnQgaXMgc2V0IHVwLlxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXJlcXVpcmUtaW1wb3J0c1xucmVxdWlyZSgnLi90ZXN0cycpO1xuIl19