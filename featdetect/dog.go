package featdetect

import (
	"container/heap"
	"math"
	"pmvs/image"
)

const (
	initialSigma float64 = 1
	sigmaStep    float64 = 1.4142135623730 // sqrt(2)
	octaveSize   int     = 4
)

// detectDogFeatures : Detects features in image using SIFT like DoG detector
func detectDogFeatures(img *image.CHWImage, mask *image.CHWImage) []*Feature {
	width := img.Width
	height := img.Height
	gridColsNum := int((width + gridSize - 1) / gridSize)
	gridRowsNum := int((height + gridSize - 1) / gridSize)

	octave := generateOctave(img)

	featMap := make([]bool, height*width, height*width)
	featGrid := make([]FeatPriorityQueue,
		gridRowsNum*gridColsNum,
		gridRowsNum*gridColsNum)

	sigma := initialSigma
	numOfFeatures := 0
	for i := 1; i < octaveSize-1; i++ {
		margin := image.GaussianMargin(sigma)
		for y := margin; y < height-margin; y++ {
			for x := margin; x < width-margin; x++ {
				if mask.At(y, x, 0) == 0 || featMap[y*width+x] {
					continue
				}
				if isLocalExtremum(octave, i, y, x) == 0 {
					continue
				}
				gridY := int(y / gridSize)
				gridX := int(x / gridSize)
				response := octave[i].At(y, x, 0)
				queue := &featGrid[gridY*gridColsNum+gridX]
				feature := NewFeature(x, y, math.Abs(float64(response)), DoG)
				heap.Push(queue, feature)
				numOfFeatures++
				featMap[y*width+x] = true
				if len(*queue) > featPerGridCell {
					heap.Pop(queue)
					numOfFeatures--
				}
			}
		}
		sigma *= sigmaStep
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
