package featdetect

import (
	"container/heap"
	"pmvs/image"
)

const (
	k           = 0.06
	harrisSigma = 3
)

func detectHarrisFeatures(img *image.CHWImage, mask *image.CHWImage) []*Feature {
	responseMap := image.HarrisCorner(image.Grayscale(img), harrisSigma, k)

	width := img.Width
	height := img.Height
	gridColsNum := int((width + gridSize - 1) / gridSize)
	gridRowsNum := int((height + gridSize - 1) / gridSize)

	featGrid := make([]FeatPriorityQueue,
		gridRowsNum*gridColsNum,
		gridRowsNum*gridColsNum)

	numOfFeatures := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			response := responseMap.At(y, x, 0)
			if response == 0 || mask.At(y, x, 0) == 0 {
				continue
			}
			gridY := int(y / gridSize)
			gridX := int(x / gridSize)
			queue := &featGrid[gridY*gridColsNum+gridX]
			feature := NewFeature(x, y, float64(response), Harris)
			heap.Push(queue, feature)
			numOfFeatures++
			if len(*queue) > featPerGridCell {
				heap.Pop(queue)
				numOfFeatures--
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
