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
import { Log } from '@tensorflow/tfjs-core';
import { unaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { logImplCPU } from '../kernel_utils/shared';
const LOG = `if (x < 0.0) return NAN;
  return log(x);`;
const LOG_PACKED = `
  vec4 result = log(x);
  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));
  result.r = isNaN.r == 1.0 ? NAN : result.r;
  result.g = isNaN.g == 1.0 ? NAN : result.g;
  result.b = isNaN.b == 1.0 ? NAN : result.b;
  result.a = isNaN.a == 1.0 ? NAN : result.a;

  return result;
`;
export const log = unaryKernelFunc({ opSnippet: LOG, packedOpSnippet: LOG_PACKED, cpuKernelImpl: logImplCPU });
export const logConfig = {
    kernelName: Log,
    backendName: 'webgl',
    kernelFunc: log
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTG9nLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL0xvZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ3BFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNuRSxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFbEQsTUFBTSxHQUFHLEdBQUc7aUJBQ0ssQ0FBQztBQUVsQixNQUFNLFVBQVUsR0FBRzs7Ozs7Ozs7O0NBU2xCLENBQUM7QUFFRixNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsZUFBZSxDQUM5QixFQUFDLFNBQVMsRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFFLFVBQVUsRUFBRSxhQUFhLEVBQUUsVUFBVSxFQUFDLENBQUMsQ0FBQztBQUU5RSxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQWlCO0lBQ3JDLFVBQVUsRUFBRSxHQUFHO0lBQ2YsV0FBVyxFQUFFLE9BQU87SUFDcEIsVUFBVSxFQUFFLEdBQXVCO0NBQ3BDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBMb2d9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge3VuYXJ5S2VybmVsRnVuY30gZnJvbSAnLi4va2VybmVsX3V0aWxzL2tlcm5lbF9mdW5jc191dGlscyc7XG5pbXBvcnQge2xvZ0ltcGxDUFV9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuXG5jb25zdCBMT0cgPSBgaWYgKHggPCAwLjApIHJldHVybiBOQU47XG4gIHJldHVybiBsb2coeCk7YDtcblxuY29uc3QgTE9HX1BBQ0tFRCA9IGBcbiAgdmVjNCByZXN1bHQgPSBsb2coeCk7XG4gIHZlYzQgaXNOYU4gPSB2ZWM0KGxlc3NUaGFuKHgsIHZlYzQoMC4wKSkpO1xuICByZXN1bHQuciA9IGlzTmFOLnIgPT0gMS4wID8gTkFOIDogcmVzdWx0LnI7XG4gIHJlc3VsdC5nID0gaXNOYU4uZyA9PSAxLjAgPyBOQU4gOiByZXN1bHQuZztcbiAgcmVzdWx0LmIgPSBpc05hTi5iID09IDEuMCA/IE5BTiA6IHJlc3VsdC5iO1xuICByZXN1bHQuYSA9IGlzTmFOLmEgPT0gMS4wID8gTkFOIDogcmVzdWx0LmE7XG5cbiAgcmV0dXJuIHJlc3VsdDtcbmA7XG5cbmV4cG9ydCBjb25zdCBsb2cgPSB1bmFyeUtlcm5lbEZ1bmMoXG4gICAge29wU25pcHBldDogTE9HLCBwYWNrZWRPcFNuaXBwZXQ6IExPR19QQUNLRUQsIGNwdUtlcm5lbEltcGw6IGxvZ0ltcGxDUFV9KTtcblxuZXhwb3J0IGNvbnN0IGxvZ0NvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBMb2csXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBsb2cgYXMge30gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==