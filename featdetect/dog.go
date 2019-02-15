package featdetect

import (
	"container/heap"
	"math"
	"pmvs/image"
)

const (
	initialSigma    float64 = 1
	sigmaStep       float64 = 1.4142135623730 // sqrt(2)
	octaveSize      int     = 4
	gridSize        int     = 32
	featPerGridCell int     = 4
)

// DetectFeatures : Detects features in image using SIFT like DoG detector
func DetectFeatures(img *image.CHWImage, mask *image.CHWImage) []*Feature {
	width := img.Width
	height := img.Height
	gridColsNum := int((width + gridSize - 1) / gridSize)
	gridRowsNum := int((height + gridSize - 1) / gridSize)

	octave := generateOctave(img)

	featMap := make([]int, height*width, height*width)
	featGrid := make([]FeatPriorityQueue, gridRowsNum*gridColsNum, gridRowsNum*gridColsNum)

	numOfFeatures := 0

	for i := 1; i < octaveSize-1; i++ {
		margin := int(math.Ceil(2 * initialSigma * math.Pow(sigmaStep, float64(i+2))))
		for y := margin; y < height-margin; y++ {
			for x := margin; x < width-margin; x++ {
				if featMap[y*width+x] != 0 {
					continue
				}
				if isLocalExtremum(octave, i, y, x) == 0 {
					continue
				}
				gridY := int(y / gridSize)
				gridX := int(x / gridSize)
				response := octave[i].At(y, x, 0)
				queue := &featGrid[gridY*gridColsNum+gridX]
				if len(*queue) >= featPerGridCell {
					if math.Abs(float64(response)) < math.Abs(float64((*queue)[0].Response)) {
						continue
					}
					heap.Pop(queue)
					numOfFeatures--
				}
				feature := NewFeature(x, y, float32(octave[i].At(y, x, 0)), DoG)

				heap.Push(queue, feature)
				numOfFeatures++
				featMap[y*width+x] = 1
			}
		}
	}

	features := make([]*Feature, 0, numOfFeatures)
	for y := 0; y < gridRowsNum; y++ {
		for x := 0; x < gridColsNum; x++ {
			features = append(features, featGrid[y*gridColsNum+x]...)
		}
	}
	return features
}

func generateOctave(img *image.CHWImage) []*image.CHWImage {
	octave := make([]*image.CHWImage, octaveSize, octaveSize)
	currentSigma := initialSigma
	imgBlurred1 := gaussianFilter(img, currentSigma)
	for i := 0; i < octaveSize; i++ {
		currentSigma *= sigmaStep
		imgBlurred2 := gaussianFilter(img, currentSigma)
		imgBlurred1.Subtract(imgBlurred2)
		octave[i] = imgBlurred1
		imgBlurred1 = imgBlurred2
	}
	return octave
}

// isLocalExtremum : Checks if the response at 'index' scale and (x, y) position is
// local extremum in the adjacent 20 neighbours in scale and space
func isLocalExtremum(octave []*image.CHWImage, index, y, x int) int {
	val := octave[index].At(y, x, 0)
	// localMin := true
	// localMax := true
	// for k := -1; k <= 1; k++ {
	// 	for j := -1; j <= 1; j++ {
	// 		for i := -1; i <= 1; i++ {
	// 			val2 := octave[index+k].At[y+j][x+i]
	// 			localMax = localMax && (val >= val2)
	// 			localMin = localMin && (val <= val2)
	// 		}
	// 		if !localMin && !localMax {
	// 			break
	// 		}
	// 	}
	// 	if !localMin && !localMax {
	// 		break
	// 	}
	// }

	// uglier but faster
	localMax := val >= octave[index].At(y, x+1, 0) && val >= octave[index].At(y, x-1, 0) &&
		val >= octave[index].At(y+1, x, 0) && val >= octave[index].At(y+1, x-1, 0) && val >= octave[index].At(y+1, x+1, 0) &&
		val >= octave[index].At(y-1, x, 0) && val >= octave[index].At(y-1, x-1, 0) && val >= octave[index].At(y-1, x+1, 0) &&
		val >= octave[index+1].At(y, x+1, 0) && val >= octave[index+1].At(y, x-1, 0) && val >= octave[index+1].At(y, x, 0) &&
		val >= octave[index+1].At(y+1, x, 0) && val >= octave[index+1].At(y+1, x-1, 0) && val >= octave[index+1].At(y+1, x+1, 0) &&
		val >= octave[index+1].At(y-1, x, 0) && val >= octave[index+1].At(y-1, x-1, 0) && val >= octave[index+1].At(y-1, x+1, 0) &&
		val >= octave[index-1].At(y, x+1, 0) && val >= octave[index-1].At(y, x-1, 0) && val >= octave[index-1].At(y, x, 0) &&
		val >= octave[index-1].At(y+1, x, 0) && val >= octave[index-1].At(y+1, x-1, 0) && val >= octave[index-1].At(y+1, x+1, 0) &&
		val >= octave[index-1].At(y-1, x, 0) && val >= octave[index-1].At(y-1, x-1, 0) && val >= octave[index-1].At(y-1, x+1, 0)
	localMin := val <= octave[index].At(y, x+1, 0) && val <= octave[index].At(y, x-1, 0) &&
		val <= octave[index].At(y+1, x, 0) && val <= octave[index].At(y+1, x-1, 0) && val <= octave[index].At(y+1, x+1, 0) &&
		val <= octave[index].At(y-1, x, 0) && val <= octave[index].At(y-1, x-1, 0) && val <= octave[index].At(y-1, x+1, 0) &&
		val <= octave[index+1].At(y, x+1, 0) && val <= octave[index+1].At(y, x-1, 0) && val <= octave[index+1].At(y, x, 0) &&
		val <= octave[index+1].At(y+1, x, 0) && val <= octave[index+1].At(y+1, x-1, 0) && val <= octave[index+1].At(y+1, x+1, 0) &&
		val <= octave[index+1].At(y-1, x, 0) && val <= octave[index+1].At(y-1, x-1, 0) && val <= octave[index+1].At(y-1, x+1, 0) &&
		val <= octave[index-1].At(y, x+1, 0) && val <= octave[index-1].At(y, x-1, 0) && val <= octave[index-1].At(y, x, 0) &&
		val <= octave[index-1].At(y+1, x, 0) && val <= octave[index-1].At(y+1, x-1, 0) && val <= octave[index-1].At(y+1, x+1, 0) &&
		val <= octave[index-1].At(y-1, x, 0) && val <= octave[index-1].At(y-1, x-1, 0) && val <= octave[index-1].At(y-1, x+1, 0)

	if localMax {
		return 1
	} else if localMin {
		return -1
	}
	return 0
}

func gaussianFilter(img *image.CHWImage, sigma float64) *image.CHWImage {
	return image.Grayscale(image.GaussianFilter(img, sigma))
}
