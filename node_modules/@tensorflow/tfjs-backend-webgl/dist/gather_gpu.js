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
import { getCoordsDataType } from './shader_compiler';
export class GatherProgram {
    constructor(aShape, outputShape) {
        this.variableNames = ['A', 'indices'];
        this.outputShape = outputShape;
        this.rank = outputShape.length;
        const dtype = getCoordsDataType(this.rank);
        const sourceCoords = getSourceCoords(aShape, 2);
        this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        setOutput(getA(${sourceCoords}));
      }
    `;
    }
}
// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape, axis) {
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < aShape.length; i++) {
        if (i === 2) {
            sourceCoords.push('int(getIndices(resRC.x, resRC.z))');
        }
        else {
            sourceCoords.push(`${currentCoords[i]}`);
        }
    }
    return sourceCoords.join();
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2F0aGVyX2dwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvZ2F0aGVyX2dwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUVwRCxNQUFNLE9BQU8sYUFBYTtJQU14QixZQUFZLE1BQWdCLEVBQUUsV0FBcUI7UUFMbkQsa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxTQUFTLENBQUMsQ0FBQztRQU0vQixJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQztRQUMvQixJQUFJLENBQUMsSUFBSSxHQUFHLFdBQVcsQ0FBQyxNQUFNLENBQUM7UUFDL0IsTUFBTSxLQUFLLEdBQUcsaUJBQWlCLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzNDLE1BQU0sWUFBWSxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFaEQsSUFBSSxDQUFDLFFBQVEsR0FBRzs7VUFFVixLQUFLO3lCQUNVLFlBQVk7O0tBRWhDLENBQUM7SUFDSixDQUFDO0NBQ0Y7QUFFRCxpRUFBaUU7QUFDakUsU0FBUyxlQUFlLENBQUMsTUFBZ0IsRUFBRSxJQUFZO0lBQ3JELE1BQU0sYUFBYSxHQUFHLENBQUMsU0FBUyxFQUFFLFNBQVMsRUFBRSxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFbkUsTUFBTSxZQUFZLEdBQUcsRUFBRSxDQUFDO0lBQ3hCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ3RDLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRTtZQUNYLFlBQVksQ0FBQyxJQUFJLENBQUMsbUNBQW1DLENBQUMsQ0FBQztTQUN4RDthQUFNO1lBQ0wsWUFBWSxDQUFDLElBQUksQ0FBQyxHQUFHLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7U0FDMUM7S0FDRjtJQUNELE9BQU8sWUFBWSxDQUFDLElBQUksRUFBRSxDQUFDO0FBQzdCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7R1BHUFVQcm9ncmFtfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtnZXRDb29yZHNEYXRhVHlwZX0gZnJvbSAnLi9zaGFkZXJfY29tcGlsZXInO1xuXG5leHBvcnQgY2xhc3MgR2F0aGVyUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ0EnLCAnaW5kaWNlcyddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG4gIHJhbms6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihhU2hhcGU6IG51bWJlcltdLCBvdXRwdXRTaGFwZTogbnVtYmVyW10pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5yYW5rID0gb3V0cHV0U2hhcGUubGVuZ3RoO1xuICAgIGNvbnN0IGR0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUodGhpcy5yYW5rKTtcbiAgICBjb25zdCBzb3VyY2VDb29yZHMgPSBnZXRTb3VyY2VDb29yZHMoYVNoYXBlLCAyKTtcblxuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgICR7ZHR5cGV9IHJlc1JDID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIHNldE91dHB1dChnZXRBKCR7c291cmNlQ29vcmRzfSkpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cblxuLy8gVGhlIGlucHV0IGFuZCBvdXRwdXQgYXJlIGFsd2F5cyBmbGF0dGVuZWQgaW50byByYW5rIDQgdGVuc29ycy5cbmZ1bmN0aW9uIGdldFNvdXJjZUNvb3JkcyhhU2hhcGU6IG51bWJlcltdLCBheGlzOiBudW1iZXIpOiBzdHJpbmcge1xuICBjb25zdCBjdXJyZW50Q29vcmRzID0gWydyZXNSQy54JywgJ3Jlc1JDLnknLCAncmVzUkMueicsICdyZXNSQy53J107XG5cbiAgY29uc3Qgc291cmNlQ29vcmRzID0gW107XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYVNoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgaWYgKGkgPT09IDIpIHtcbiAgICAgIHNvdXJjZUNvb3Jkcy5wdXNoKCdpbnQoZ2V0SW5kaWNlcyhyZXNSQy54LCByZXNSQy56KSknKTtcbiAgICB9IGVsc2Uge1xuICAgICAgc291cmNlQ29vcmRzLnB1c2goYCR7Y3VycmVudENvb3Jkc1tpXX1gKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHNvdXJjZUNvb3Jkcy5qb2luKCk7XG59XG4iXX0=